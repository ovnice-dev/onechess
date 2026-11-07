"""Entraineur JAX (TPU Kaggle, quota separe des GPU) pour l'architecture du reseau final.

Meme reseau, memes donnees, meme perte que tools/train.py (verifie pas a pas :
tools/train_jax.py --verifier) : 1re couche a experts par materiel (mat4 / mat8)
avec tronc commun (factoriseur), CReLU, sorties par nombre de pieces, perte
(sigmoid(pred) - cible)^2, Adam, pas en cosinus par epoque, poids bornes a
+-1,98. L'export passe par tools/train.py : meme fichier .nnue, meme moteur.
Les poids s'echangent avec train.py (--init / --greffe .pt, --save-pt) : on
peut entrainer sur GPU, continuer sur TPU, et inversement.

    python tools/train_jax.py --stream "/tmp/donnees/*.bin" --hidden 1024 --buckets mat4 \
        --out-buckets 4 --epochs 60 --epoch-files 60 --chunk-files 10 \
        --checkpoint final.pkl --out final.nnue
"""
import argparse
import glob
import math
import os
import pickle
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

ICI = os.path.dirname(os.path.abspath(__file__))
BASE, QA, SCALE = 768, 255, 400
CLIP = 1.98


# --------------------------------------------------------------------------
# Decodage des enregistrements de 32 octets (identique a RawData.batch)
# --------------------------------------------------------------------------

def decoder(r, ob_n, lam):
    r = r.astype(jnp.int32)
    B = r.shape[0]
    bits = ((r[:, 0:8, None] >> jnp.arange(8)) & 1).reshape(B, 64)
    counts = bits.sum(1)
    pcs = r[:, 8:24]
    nib = jnp.stack([pcs & 0x0F, pcs >> 4], axis=2).reshape(B, 32)
    codes = jnp.where(jnp.arange(32)[None] < counts[:, None], nib, 12)
    sqs = jnp.argsort(-bits, axis=1, stable=True)[:, :32]     # cases occupees, ordre croissant
    sc = r[:, 24] | (r[:, 25] << 8)
    sc = jnp.where(sc >= 32768, sc - 65536, sc).astype(jnp.float32)
    stm = r[:, 26]
    res = r[:, 27].astype(jnp.float32) / 2.0
    sign = 1.0 - 2.0 * stm.astype(jnp.float32)
    sc = sc * sign
    res = jnp.where(stm == 0, res, 1.0 - res)
    tg = lam * jax.nn.sigmoid(sc / SCALE) + (1.0 - lam) * res
    ob = jnp.minimum((counts - 2) // (32 // ob_n), ob_n - 1)
    return codes, sqs, stm, ob, sc, tg


def indices(codes, sqs, persp, nb):
    valid = codes < 12
    colour = codes // 6
    ptype = codes % 6
    pidx = (colour != persp).astype(jnp.int32) * 6 + ptype
    rel = sqs ^ (56 * persp)
    base = pidx * 64 + rel
    n = valid.sum(1, keepdims=True)
    feat = jnp.clip((n - 2) * nb // 32, 0, nb - 1) * BASE + base
    return jnp.where(valid, feat, nb * BASE), jnp.where(valid, base, BASE)


def predire(p, r, nb, ob_n, lam):
    codes, sqs, stm, ob, sc, tg = decoder(r, ob_n, lam)
    fw, bw = indices(codes, sqs, 0, nb)
    fb, bb = indices(codes, sqs, 1, nb)
    aw = p["ft"][fw].sum(1) + p["ft_v"][bw].sum(1) + p["ft_bias"]
    ab = p["ft"][fb].sum(1) + p["ft_v"][bb].sum(1) + p["ft_bias"]
    s = stm[:, None] == 0
    us, them = jnp.where(s, aw, ab), jnp.where(s, ab, aw)
    x = jnp.concatenate([jnp.clip(us, 0, 1), jnp.clip(them, 0, 1)], axis=1)
    y = x @ p["out_w"].T + p["out_b"]
    pred = jnp.take_along_axis(y, ob[:, None], axis=1)[:, 0]
    return pred, tg, sc, jnp.maximum(jnp.abs(aw).max(), jnp.abs(ab).max())


def perte(p, r, nb, ob_n, lam):
    pred, tg, _, _ = predire(p, r, nb, ob_n, lam)
    return jnp.mean((jax.nn.sigmoid(pred) - tg) ** 2)


def borner(p, nb):
    p = dict(p)
    p["ft"] = jnp.clip(p["ft"], -CLIP, CLIP).at[nb * BASE].set(0.0)
    p["ft_v"] = jnp.clip(p["ft_v"], -CLIP, CLIP).at[BASE].set(0.0)
    p["ft_bias"] = jnp.clip(p["ft_bias"], -CLIP, CLIP)
    p["out_w"] = jnp.clip(p["out_w"], -CLIP, CLIP)
    return p


# --------------------------------------------------------------------------
# Poids : creation, echange avec train.py (PyTorch)
# --------------------------------------------------------------------------

def nouveaux(nb, hidden, ob_n, graine):
    k = jax.random.split(jax.random.PRNGKey(graine), 3)
    n_in = nb * BASE
    return borner({
        "ft": 0.01 * jax.random.normal(k[0], (n_in + 1, hidden)),
        "ft_v": 0.01 * jax.random.normal(k[1], (BASE + 1, hidden)),
        "ft_bias": jnp.zeros(hidden),
        "out_w": 0.05 * jax.random.normal(k[2], (ob_n, 2 * hidden)),
        "out_b": jnp.zeros(ob_n),
    }, nb)


NOMS = {"ft": "ft.weight", "ft_v": "ft_v.weight", "ft_bias": "ft_bias", "out_w": "out.weight", "out_b": "out.bias"}


def depuis_torch(sd, p, nb):
    """Charge un state_dict de train.py ; experts 4 -> 8 comme train.py --greffe."""
    p = dict(p)
    for k, nom in NOMS.items():
        w = np.asarray(sd[nom].cpu().numpy() if hasattr(sd[nom], "cpu") else sd[nom], dtype=np.float32)
        if w.shape == p[k].shape:
            p[k] = jnp.asarray(w)
        elif k == "ft" and w.shape[1] == p[k].shape[1]:
            nb_old = (w.shape[0] - 1) // BASE
            v = np.zeros(p[k].shape, np.float32)
            for b in range(nb):
                o = b * nb_old // nb
                v[b * BASE:(b + 1) * BASE] = w[o * BASE:(o + 1) * BASE]
            p[k] = jnp.asarray(v)
            print(f"greffe : experts {nb_old} -> {nb}")
        else:
            raise SystemExit(f"greffe impossible pour {nom} : {w.shape} -> {p[k].shape}")
    return borner(p, nb)


def vers_torch(p):
    import torch
    return {nom: torch.from_numpy(np.asarray(p[k], dtype=np.float32)) for k, nom in NOMS.items()}


def exporter(p, nb, hidden, ob_n, chemin, empirical, save_pt=""):
    import torch
    sys.path.insert(0, ICI)
    import train as T
    m = T.Perspective(nb * BASE, hidden, ob_n, factorizer=True)
    m.load_state_dict(vers_torch(p), strict=False)
    with torch.no_grad():   # couches inutilisees par ce reseau : sans effet sur l'export
        pass
    fmap = T.FeatureMap("mat8" if nb == 8 else "mat4", 0, "cpu")
    T.export(m, fmap, chemin, hidden, empirical)
    if save_pt:
        torch.save(m.state_dict(), save_pt)
        print("poids flottants ecrits :", save_pt)


# --------------------------------------------------------------------------
# Entrainement
# --------------------------------------------------------------------------

def fichiers_epoque(n, k, epoch, seed):
    start, out = (epoch - 1) * k, []
    while len(out) < k:
        cycle, off = divmod(start, n)
        perm = np.random.default_rng(seed * 7919 + cycle).permutation(n).tolist()
        take = min(k - len(out), n - off)
        out += perm[off:off + take]
        start += take
    return out


def charger(fichiers):
    return np.concatenate([np.fromfile(f, dtype=np.uint8).reshape(-1, 32) for f in fichiers])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=False, default="")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--buckets", default="mat4", choices=["mat4", "mat8"])
    ap.add_argument("--out-buckets", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--epoch-files", type=int, default=0)
    ap.add_argument("--chunk-files", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-eval", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", type=int, default=5_000_000)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out", default="reseau.nnue")
    ap.add_argument("--save-pt", default="")
    ap.add_argument("--init", default="", help="poids .pt de train.py (ou greffe 4 -> 8 experts)")
    ap.add_argument("--limite-s", type=float, default=0, help="arret propre apres ce temps (session)")
    ap.add_argument("--max-pas", type=int, default=0, help="essai de vitesse : arret apres N pas")
    ap.add_argument("--verifier", action="store_true", help="comparaison pas a pas avec train.py")
    a = ap.parse_args()
    if a.verifier:
        return verifier(a)
    t0 = time.time()

    nb, ob_n = (8 if a.buckets == "mat8" else 4), a.out_buckets
    devs = jax.devices()
    nd = len(devs)
    print("appareils :", devs, flush=True)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    mesh = Mesh(np.array(devs), ("b",))
    rep, lot = NamedSharding(mesh, P()), NamedSharding(mesh, P("b"))
    B = a.batch - a.batch % nd

    files = sorted(glob.glob(a.stream))
    val = jax.device_put(charger(files[-1:])[:a.val], NamedSharding(mesh, P()))
    train = files[:-1]
    print(f"{len(train)} fichiers d'entrainement, {val.shape[0]:,} de validation", flush=True)

    p = nouveaux(nb, a.hidden, ob_n, a.seed)
    if a.init:
        import torch
        g = torch.load(a.init, map_location="cpu", weights_only=False)
        if isinstance(g, dict) and "model" in g and "epoch" in g:
            g = g.get("best_state") or g["model"]
        p = depuis_torch(g, p, nb)
        print("poids de depart :", a.init, flush=True)
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=a.lr, b1=0.9, b2=0.999, eps=1e-8)
    st = opt.init(p)
    first, best, best_p = 1, (1e9, 1e9, 0), None
    if a.checkpoint and os.path.exists(a.checkpoint):
        ck = pickle.load(open(a.checkpoint, "rb"))
        p, st = jax.tree.map(jnp.asarray, ck["p"]), jax.tree.map(jnp.asarray, ck["st"])
        first, best, best_p = ck["epoch"] + 1, tuple(ck["best"]), ck["best_p"]
        print(f"reprise : epoque {first}/{a.epochs}, meilleure {best[1]:.1f} cp", flush=True)
    p, st = jax.device_put(p, rep), jax.device_put(st, rep)

    @jax.jit
    def pas(p, st, r):
        l, g = jax.value_and_grad(perte)(p, r, nb, ob_n, a.lambda_eval)
        u, st = opt.update(g, st, p)
        return borner(optax.apply_updates(p, u), nb), st, l

    @jax.jit
    def evaluer(p, r):
        pred, tg, sc, mx = predire(p, r, nb, ob_n, a.lambda_eval)
        return ((jax.nn.sigmoid(pred) - tg) ** 2).sum(), jnp.abs(pred * SCALE - sc).sum(), mx

    def validation(p):
        sl = se = 0.0
        mx = 0.0
        for i in range(0, val.shape[0], 65536):
            l, e, m = evaluer(p, val[i:i + 65536])
            sl, se, mx = sl + float(l), se + float(e), max(mx, float(m))
        return sl / val.shape[0], se / val.shape[0], mx

    for epoch in range(first, a.epochs + 1):
        eta_min = a.lr / 30
        lr = eta_min + (a.lr - eta_min) * (1 + math.cos(math.pi * (epoch - 1) / a.epochs)) / 2
        st.hyperparams["learning_rate"] = jnp.asarray(lr)
        te, tot, n_tot, n_pas, n_pos = time.time(), 0.0, 0, 0, 0
        if a.epoch_files:
            ordre = fichiers_epoque(len(train), a.epoch_files, epoch, a.seed)
        else:
            ordre = np.random.default_rng(a.seed * 104729 + epoch).permutation(len(train)).tolist()
        rng = np.random.default_rng(a.seed * 31 + epoch)
        for g in range(0, len(ordre), a.chunk_files):
            recs = charger([train[i] for i in ordre[g:g + a.chunk_files]])
            perm = rng.permutation(len(recs))
            for k in range(len(recs) // B):
                r = jax.device_put(recs[perm[k * B:(k + 1) * B]], lot)
                p, st, l = pas(p, st, r)
                n_pas += 1
                n_pos += B
                if n_pas % 50 == 1:            # echantillon : lire la perte force une synchronisation
                    tot, n_tot = tot + float(l), n_tot + 1
                if a.max_pas and n_pas >= a.max_pas:
                    jax.block_until_ready(p)
                    dt = time.time() - te
                    print(f"ESSAI DE VITESSE : {n_pos:,} positions en {dt:.1f} s = {n_pos / dt:,.0f} positions/s "
                          f"({nd} appareils, lots de {B})", flush=True)
                    return
            if a.limite_s and time.time() - t0 > a.limite_s:
                break
        vl, ce, mx = validation(p)
        if ce < best[1]:
            best, best_p = (vl, ce, epoch), jax.tree.map(np.asarray, p)
        dt = time.time() - te
        print(f"  epoque {epoch:3d}/{a.epochs}  perte {tot / max(1, n_tot):.5f}  val {vl:.5f}  erreur {ce:6.1f} cp  "
              f"({dt:.0f} s, {n_pos / dt:,.0f} positions/s)", flush=True)
        if a.checkpoint:
            tmp = a.checkpoint + ".tmp"
            pickle.dump({"p": jax.tree.map(np.asarray, p), "st": jax.tree.map(np.asarray, st), "epoch": epoch,
                         "best": list(best), "best_p": best_p}, open(tmp, "wb"))
            os.replace(tmp, a.checkpoint)
        if a.limite_s and time.time() - t0 > a.limite_s:
            print("===== LIMITE DE SESSION : arret, a reprendre dans la session suivante", flush=True)
            return
    _, _, mx = validation(jax.tree.map(jnp.asarray, best_p))
    print(f"export de l'epoque {best[2]} ({best[1]:.1f} cp)", flush=True)
    exporter(best_p, nb, a.hidden, ob_n, a.out, int(round(mx * QA)), a.save_pt)


# --------------------------------------------------------------------------
# Verification pas a pas contre train.py (memes poids, meme lot)
# --------------------------------------------------------------------------

def verifier(a):
    import torch
    sys.path.insert(0, ICI)
    import train as T
    nb, ob_n, H = (8 if a.buckets == "mat8" else 4), a.out_buckets, a.hidden
    files = sorted(glob.glob(a.stream))
    recs = np.fromfile(files[0], dtype=np.uint8).reshape(-1, 32)[:4096]
    torch.manual_seed(0)
    m = T.Perspective(nb * BASE, H, ob_n, factorizer=True)
    m.clip_weights()
    fmap = T.FeatureMap(a.buckets, 0, "cpu")
    d = T.RawData(recs, "cpu", ob_n, a.lambda_eval)
    c, sq, stm, ob, sc, tg = d.batch(torch.arange(len(recs)))
    iw = fmap.indices(c, sq, 0)
    ib = fmap.indices(c, sq, 1)
    pred_t = m(iw, ib, stm, ob)
    lt = ((torch.sigmoid(pred_t) - tg) ** 2).mean()
    lt.backward()
    p = depuis_torch({k: v.detach() for k, v in m.state_dict().items()}, nouveaux(nb, H, ob_n, 0), nb)
    lj, gj = jax.value_and_grad(perte)(p, jnp.asarray(recs), nb, ob_n, a.lambda_eval)
    pj = predire(p, jnp.asarray(recs), nb, ob_n, a.lambda_eval)[0]
    print(f"perte : torch {lt.item():.8f}, jax {float(lj):.8f}")
    print(f"prediction : ecart max {np.abs(np.asarray(pj) - pred_t.detach().numpy()).max():.2e}")
    for k, nom in NOMS.items():
        gt = dict(m.named_parameters())[nom].grad.numpy()
        gk = np.asarray(gj[k])
        if k in ("ft", "ft_v"):        # ligne de remplissage : sans effet (remise a zero a chaque pas)
            gt, gk = gt[:-1], gk[:-1]
        print(f"gradient {nom:12s} : ecart max {np.abs(gk - gt).max():.2e} "
              f"(echelle {np.abs(gt).max():.2e})")


if __name__ == "__main__":
    main()
