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
BASE, SCALE = 768, 400
QA = int(os.environ.get('ONE_QA', 255))   # ONE_QA=127 avec --ft-int8 1 : 1re couche int8 dans le moteur
QAT = False   # --ft-int8 1 : poids fusionnes bornes a 127/QA et arrondis a l'aller (entrainement conscient de la quantification)
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


def decoder_cases(r, ob_n, lam):
    """Meme decodage, sans tri (lent sur TPU) : chaque case occupee lit son
    quartet par son rang parmi les cases occupees. Rend des tableaux (B, 64),
    une colonne par case (code 12 = case vide)."""
    r = r.astype(jnp.int32)
    B = r.shape[0]
    bits = ((r[:, 0:8, None] >> jnp.arange(8)) & 1).reshape(B, 64)
    counts = bits.sum(1)
    pcs = r[:, 8:24]
    nib = jnp.stack([pcs & 0x0F, pcs >> 4], axis=2).reshape(B, 32)
    rang = jnp.clip(jnp.cumsum(bits, axis=1) - 1, 0, 31)
    codes = jnp.where(bits == 1, jnp.take_along_axis(nib, rang, axis=1), 12)
    sqs = jnp.broadcast_to(jnp.arange(64), (B, 64))
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


def multi_chaud(idx, n):
    """(B, k) indices -> (B, n) comptes : la 1re couche devient un produit de matrices denses (TPU)."""
    B = idx.shape[0]
    return jnp.zeros((B, n), jnp.float32).at[jnp.arange(B)[:, None], idx].add(1.0)


DENSE = False      # --ft-dense : 1re couche en produits de matrices (TPU)
# --menaces 1 (phase 10, C) : entrees "menaces completes" (59 808, tools/menaces_completes.py),
# table th de (DIM + 1) x hidden ajoutee a la 1re couche ; la ligne DIM (remplissage) reste nulle.
# Indices calcules sur l'hote (indexeur C++) et passes avec le lot : r = (positions, iw, ib).
MENACES = False
LARG_MEN = 0        # --menaces-largeur : neurones alimentes par les menaces (0 = tous) ; bloc etroit = moins de memoire lue
ENNEMIS = False     # --menaces-ennemis 1 : sans les defenses (cible du meme camp)
K_MEN = 64
DIM_MEN = 59808
CLIP_TH = 127 / QA   # poids des menaces en int8 dans le moteur (comme Stockfish), meme echelle QA


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


def entrees_denses(r, nb, ob_n, lam):
    """Matrices des pieces (B, nb*768) et (B, 768) pour chaque perspective, sans
    tri, sans lecture ni ecriture dispersee (lentes sur TPU) : comparaisons et
    produits seulement. Meme contenu que les indices de indices()."""
    r = r.astype(jnp.int32)
    B = r.shape[0]
    bits = ((r[:, 0:8, None] >> jnp.arange(8)) & 1).reshape(B, 64)
    counts = bits.sum(1)
    pcs = r[:, 8:24]
    nib = jnp.stack([pcs & 0x0F, pcs >> 4], axis=2).reshape(B, 32).astype(jnp.float32)
    rang = jnp.cumsum(bits, axis=1) - 1                                    # rang de chaque case occupee
    sel = (rang[:, :, None] == jnp.arange(32)[None, None, :]) & (bits[:, :, None] == 1)
    codes = jnp.einsum("bsk,bk->bs", sel.astype(jnp.float32), nib)        # quartet de chaque case
    codes = jnp.where(bits == 1, codes, 12.0)
    xw = (codes[:, :, None] == jnp.arange(12, dtype=jnp.float32)[None, None, :]).astype(jnp.float32)
    xw = xw.transpose(0, 2, 1)                                             # (B, 12 pieces, 64 cases)
    # Vue des noirs : couleurs echangees, echiquier retourne (case ^ 56 = rangee inversee)
    xb = jnp.concatenate([xw[:, 6:12], xw[:, 0:6]], axis=1).reshape(B, 12, 8, 8)[:, :, ::-1, :]
    xw, xb = xw.reshape(B, 768), xb.reshape(B, 768)
    bande = jnp.clip((counts - 2) * nb // 32, 0, nb - 1)
    ob1 = (bande[:, None] == jnp.arange(nb)[None, :]).astype(jnp.float32)
    fw = (ob1[:, :, None] * xw[:, None, :]).reshape(B, nb * BASE)
    fb = (ob1[:, :, None] * xb[:, None, :]).reshape(B, nb * BASE)
    sc = r[:, 24] | (r[:, 25] << 8)
    sc = jnp.where(sc >= 32768, sc - 65536, sc).astype(jnp.float32)
    stm = r[:, 26]
    res = r[:, 27].astype(jnp.float32) / 2.0
    sign = 1.0 - 2.0 * stm.astype(jnp.float32)
    sc = sc * sign
    res = jnp.where(stm == 0, res, 1.0 - res)
    tg = lam * jax.nn.sigmoid(sc / SCALE) + (1.0 - lam) * res
    ob = jnp.minimum((counts - 2) // (32 // ob_n), ob_n - 1)
    return fw, fb, xw, xb, stm, ob, sc, tg


def quantifier(p, nb):
    """QAT : a l'aller, poids fusionnes (bucket + tronc) et biais arrondis comme dans le moteur
    (round(w * QA) / QA) ; au retour, gradient inchange (estimateur droit)."""
    n, h = nb * BASE, p["ft_bias"].shape[0]
    m = p["ft"][:n].reshape(nb, BASE, h) + p["ft_v"][:BASE][None]
    dm = jax.lax.stop_gradient(jnp.round(m * QA) / QA - m).reshape(n, h)
    q = dict(p)
    q["ft"] = p["ft"].at[:n].add(dm)
    b = p["ft_bias"]
    q["ft_bias"] = b + jax.lax.stop_gradient(jnp.round(b * QA) / QA - b)
    return q


def predire(p, r, nb, ob_n, lam):
    if QAT:
        p = quantifier(p, nb)
    if MENACES:
        r, iw, ib = r
        tw, tb = p["th"][iw].sum(1), p["th"][ib].sum(1)
        h = p["ft_bias"].shape[0]
        if tw.shape[1] < h:   # bloc etroit : les menaces n'alimentent que les premiers neurones
            tw = jnp.pad(tw, ((0, 0), (0, h - tw.shape[1])))
            tb = jnp.pad(tb, ((0, 0), (0, h - tb.shape[1])))
    else:
        tw = tb = 0.0
    if DENSE:
        fw, fb, xw, xb, stm, ob, sc, tg = entrees_denses(r, nb, ob_n, lam)
        hp = jax.lax.Precision.HIGHEST
        n = nb * BASE
        aw = (jnp.matmul(fw, p["ft"][:n], precision=hp) + jnp.matmul(xw, p["ft_v"][:BASE], precision=hp)
              + p["ft_bias"] + tw)
        ab = (jnp.matmul(fb, p["ft"][:n], precision=hp) + jnp.matmul(xb, p["ft_v"][:BASE], precision=hp)
              + p["ft_bias"] + tb)
        s = stm[:, None] == 0
        us, them = jnp.where(s, aw, ab), jnp.where(s, ab, aw)
        x = jnp.concatenate([jnp.clip(us, 0, 1), jnp.clip(them, 0, 1)], axis=1)
        y = x @ p["out_w"].T + p["out_b"]
        pred = jnp.take_along_axis(y, ob[:, None], axis=1)[:, 0]
        return pred, tg, sc, jnp.maximum(jnp.abs(aw).max(), jnp.abs(ab).max())
    if False:
        codes, sqs, stm, ob, sc, tg = decoder_cases(r, ob_n, lam)
    else:
        codes, sqs, stm, ob, sc, tg = decoder(r, ob_n, lam)
    fw, bw = indices(codes, sqs, 0, nb)
    fb, bb = indices(codes, sqs, 1, nb)
    if DENSE:
        hp = jax.lax.Precision.HIGHEST
        nf, nv = p["ft"].shape[0], p["ft_v"].shape[0]
        aw = (jnp.matmul(multi_chaud(fw, nf), p["ft"], precision=hp)
              + jnp.matmul(multi_chaud(bw, nv), p["ft_v"], precision=hp) + p["ft_bias"])
        ab = (jnp.matmul(multi_chaud(fb, nf), p["ft"], precision=hp)
              + jnp.matmul(multi_chaud(bb, nv), p["ft_v"], precision=hp) + p["ft_bias"])
    else:
        aw = p["ft"][fw].sum(1) + p["ft_v"][bw].sum(1) + p["ft_bias"] + tw
        ab = p["ft"][fb].sum(1) + p["ft_v"][bb].sum(1) + p["ft_bias"] + tb
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
    if "th" in p:
        p["th"] = jnp.clip(p["th"], -CLIP_TH, CLIP_TH).at[DIM_MEN].set(0.0)
    if QAT:   # poids FUSIONNES bornes a 127/QA (int8 dans le moteur), ecart reporte sur la part du bucket
        n, h, c = nb * BASE, p["ft_bias"].shape[0], 127 / QA
        v = p["ft_v"][:BASE][None]
        m = jnp.clip(p["ft"][:n].reshape(nb, BASE, h) + v, -c, c) - v
        p["ft"] = p["ft"].at[:n].set(m.reshape(n, h))
    return p


# --------------------------------------------------------------------------
# Poids : creation, echange avec train.py (PyTorch)
# --------------------------------------------------------------------------

def nouveaux(nb, hidden, ob_n, graine):
    k = jax.random.split(jax.random.PRNGKey(graine), 3)
    n_in = nb * BASE
    th = {"th": jnp.zeros((DIM_MEN + 1, LARG_MEN or hidden))} if MENACES else {}   # a zero : greffe sans effet au depart
    return borner({**th,
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


FULLTHREAT_FLAG = 4096   # moteur : bit 12 du champ activation (nnue.h)


def ecrire_menaces(chemin, th, nb, hidden, ennemis=False):
    """Insere les menaces completes apres les biais de la 1re couche d'un .nnue ecrit par
    train.export et leve le bit FULLTHREAT_FLAG : largeur du bloc (uint32), options
    (uint32, bit 0 = sans defenses), puis la table int8 DIM x largeur."""
    import struct
    raw = bytearray(open(chemin, "rb").read())
    act = struct.unpack_from("<I", raw, 20)[0]
    assert struct.unpack_from("<I", raw, 4)[0] == hidden and struct.unpack_from("<I", raw, 8)[0] == nb
    struct.pack_into("<I", raw, 20, act | FULLTHREAT_FLAG)
    off = 24 + 64 + nb * BASE * hidden * 2 + hidden * 2
    q = np.clip(np.round(np.asarray(th[:DIM_MEN], dtype=np.float64) * QA), -127, 127).astype(np.int8)
    tete = struct.pack("<II", q.shape[1], int(ennemis))
    open(chemin, "wb").write(bytes(raw[:off]) + tete + q.tobytes() + bytes(raw[off:]))
    print(f"menaces completes inserees : {q.shape[0]} x {q.shape[1]} int8 ({q.nbytes / 1e6:.0f} Mo)", flush=True)


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
# Points de reprise echangeables avec train.py (GPU <-> TPU, 7-11)
# --------------------------------------------------------------------------

ORDRE_TORCH = ["ft_bias", "ft", "ft_v", "out_w", "out_b"]     # ordre de model.parameters() dans train.py


def depuis_ckpt_torch(chemin, p, opt, nb):
    """Point de reprise de train.py -> (poids, etat Adam, epoque, meilleure, poids de la meilleure)."""
    import torch
    ck = torch.load(chemin, map_location="cpu", weights_only=False)
    p = depuis_torch(ck["model"], p, nb)
    st = opt.init(p)
    etat = ck["opt"]["state"]
    mu, nu = dict(st.inner_state[0].mu), dict(st.inner_state[0].nu)
    pas = 0
    for i, k in enumerate(ORDRE_TORCH):
        if i in etat:
            mu[k] = jnp.asarray(etat[i]["exp_avg"].numpy())
            nu[k] = jnp.asarray(etat[i]["exp_avg_sq"].numpy())
            pas = int(float(etat[i]["step"]))
    adam = st.inner_state[0]._replace(count=jnp.asarray(pas, jnp.int32), mu=mu, nu=nu)
    st = st._replace(count=jnp.asarray(pas, jnp.int32), inner_state=(adam,) + tuple(st.inner_state[1:]))
    bp = depuis_torch(ck["best_state"], p, nb) if ck.get("best_state") else None
    print(f"point de reprise PyTorch {chemin} : epoque {ck['epoch']}, {pas} pas d'Adam", flush=True)
    return p, st, ck["epoch"], tuple(ck["best"]), (jax.tree.map(np.asarray, bp) if bp is not None else None)


def vers_ckpt_torch(chemin, p, st, epoch, best, best_p, nb, hidden, ob_n, lr, epochs):
    """Ecrit un point de reprise au format de train.py (que train.py reprend sur GPU)."""
    import torch
    sys.path.insert(0, ICI)
    import train as T
    m = T.Perspective(nb * BASE, hidden, ob_n, factorizer=True)
    m.load_state_dict(vers_torch(p), strict=False)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 30)
    adam = st.inner_state[0]
    pas = float(np.asarray(adam.count))
    for prm, k in zip(m.parameters(), ORDRE_TORCH):
        opt.state[prm] = {"step": torch.tensor(pas),
                          "exp_avg": torch.from_numpy(np.asarray(adam.mu[k], np.float32).copy()),
                          "exp_avg_sq": torch.from_numpy(np.asarray(adam.nu[k], np.float32).copy())}
    for _ in range(epoch):
        sched.step()
    bs = {n: v.clone() for n, v in m.state_dict().items()}
    if best_p is not None:
        m2 = T.Perspective(nb * BASE, hidden, ob_n, factorizer=True)
        m2.load_state_dict(vers_torch(best_p), strict=False)
        bs = m2.state_dict()
    tmp = chemin + ".tmp"
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                "best": list(best), "best_state": bs, "rng": torch.get_rng_state(), "rng_cuda": None}, tmp)
    os.replace(tmp, chemin)


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


def avec_menaces(recs):
    """Lot de positions -> (positions, indices blancs, indices noirs) si --menaces, sinon le lot."""
    if not MENACES:
        return recs
    sys.path.insert(0, ICI)
    import menaces_completes as M
    iw, ib, m = M.lot(recs, K_MEN, ennemis=ENNEMIS)
    if m > K_MEN:
        print(f"[menaces] {m} paires actives > {K_MEN} : tronque", flush=True)
    return recs, iw.astype(np.int32), ib.astype(np.int32)


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
    ap.add_argument("--reprise-torch", default="", help="point de reprise .ckpt de train.py (session GPU precedente)")
    ap.add_argument("--ckpt-torch", default="", help="ecrire aussi a chaque epoque un point de reprise pour train.py")
    ap.add_argument("--limite-s", type=float, default=0, help="arret propre apres ce temps (session)")
    ap.add_argument("--max-pas", type=int, default=0, help="essai de vitesse : arret apres N pas")
    ap.add_argument("--verifier", action="store_true", help="comparaison pas a pas avec train.py")
    ap.add_argument("--ft-dense", type=int, default=0, help="1 = 1re couche en produits de matrices denses (TPU)")
    ap.add_argument("--menaces", type=int, default=0, help="1 = entrees menaces completes (phase 10, C)")
    ap.add_argument("--menaces-largeur", type=int, default=0, help="neurones alimentes par les menaces (0 = tous)")
    ap.add_argument("--menaces-ennemis", type=int, default=0, help="1 = sans les defenses")
    ap.add_argument("--ft-int8", type=int, default=0,
                    help="1 = QAT : 1re couche int8 dans le moteur (lancer avec ONE_QA=127, moteur -DONE_QA=127 -DONE_FT_INT8)")
    a = ap.parse_args()
    global DENSE, MENACES, LARG_MEN, ENNEMIS, QAT
    QAT = bool(a.ft_int8)
    if QAT:
        print(f"QAT : 1re couche int8, QA = {QA}, poids fusionnes bornes a {127 / QA:.3f}", flush=True)
    LARG_MEN, ENNEMIS = a.menaces_largeur, bool(a.menaces_ennemis)
    DENSE = bool(a.ft_dense)
    MENACES = bool(a.menaces)
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
    val = jax.device_put(avec_menaces(charger(files[-1:])[:a.val]), NamedSharding(mesh, P()))
    train = files[:-1]
    NVAL = (val[0] if MENACES else val).shape[0]
    print(f"{len(train)} fichiers d'entrainement, {NVAL:,} de validation", flush=True)

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
    if a.reprise_torch and not (a.checkpoint and os.path.exists(a.checkpoint)):
        p, st, e0, best, best_p = depuis_ckpt_torch(a.reprise_torch, p, opt, nb)
        first = e0 + 1
    elif a.checkpoint and os.path.exists(a.checkpoint):
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
        for i in range(0, NVAL, 65536):
            l, e, m = evaluer(p, jax.tree.map(lambda v: v[i:i + 65536], val))
            sl, se, mx = sl + float(l), se + float(e), max(mx, float(m))
        return sl / NVAL, se / NVAL, mx

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
                r = jax.device_put(avec_menaces(recs[perm[k * B:(k + 1) * B]]), lot)
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
        if a.ckpt_torch and not MENACES:
            vers_ckpt_torch(a.ckpt_torch, jax.tree.map(np.asarray, p), jax.tree.map(np.asarray, st), epoch, best,
                            best_p, nb, a.hidden, ob_n, a.lr, a.epochs)
        if a.limite_s and time.time() - t0 > a.limite_s:
            print("===== LIMITE DE SESSION : arret, a reprendre dans la session suivante", flush=True)
            return
    _, _, mx = validation(jax.tree.map(jnp.asarray, best_p))
    print(f"export de l'epoque {best[2]} ({best[1]:.1f} cp)", flush=True)
    exporter(best_p, nb, a.hidden, ob_n, a.out, int(round(mx * QA)), a.save_pt)
    if MENACES:
        ecrire_menaces(a.out, best_p["th"], nb, a.hidden, ENNEMIS)
        np.save(a.out + ".menaces.npy", np.asarray(best_p["th"], dtype=np.float32))


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
