#!/usr/bin/env python3
"""Entrainement du reseau NNUE de Onechess.

Architecture : (768 -> HL) x2 perspectives -> 1.

Les 768 entrees sont 12 types de pieces x 64 cases, vues depuis le camp
considere. Chaque camp a son accumulateur ; la couche de sortie lit d'abord
celui du camp au trait. Les deux perspectives partagent les memes poids, ce
qui divise par deux ce que le reseau a a apprendre.

Cible : un melange entre le score de Stockfish et le resultat de la partie.
Le score dit ou en est la position, le resultat dit ce qu'elle devient ; un
reseau entraine sur le seul score joue bien mais evalue mal les positions
bloquees, un reseau entraine sur le seul resultat est trop bruite.

Contraintes de quantification appliquees pendant l'entrainement, et non
apres : un poids trop grand ferait deborder l'accumulateur en int16, et
l'ecretage a posteriori couterait de la force sans qu'on sache combien.
"""

import argparse
import glob
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE  = 768
PAD_INDEX   = INPUT_SIZE          # ligne nulle, exclue de la somme
QA, QB      = 255, 64
SCALE       = 400
MAGIC       = 0x314E4E4F

MAX_PIECES  = 32


# --------------------------------------------------------------------------
# Lecture des enregistrements binaires
# --------------------------------------------------------------------------

def decode_shards(pattern, limit=0, cache=None):
    """Rend (idx, score, stm, result) ; idx est (N, 32) int16 remplie a 768."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        print(f"cache lu : {cache}  ({len(z['score']):,} positions)")
        return z["idx"], z["score"], z["stm"], z["result"]

    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"aucun fichier ne correspond a {pattern}")

    chunks = []
    total = 0
    for f in files:
        raw = np.fromfile(f, dtype=np.uint8)
        n = len(raw) // 32
        if n == 0:
            continue
        chunks.append(raw[: n * 32].reshape(n, 32))
        total += n
        if limit and total >= limit:
            break

    recs = np.concatenate(chunks, axis=0)
    if limit:
        recs = recs[:limit]
    n = len(recs)
    print(f"{n:,} positions lues dans {len(chunks)} fichiers")

    score  = recs[:, 24:26].copy().view(np.int16).ravel().astype(np.float32)
    stm    = recs[:, 26].astype(np.int64)
    result = recs[:, 27].astype(np.float32) / 2.0        # 0, 0.5, 1

    idx = np.full((n, MAX_PIECES), PAD_INDEX, dtype=np.int16)
    sq64 = np.arange(64, dtype=np.int16)
    bitpos = np.arange(64, dtype=np.uint64)

    # Par tranches : les tableaux intermediaires sont en (taille, 64) et
    # argsort rend des int64, ce qui saturerait la memoire sur 12 M de lignes.
    CHUNK = 250_000
    t_dec = time.time()
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        r = recs[a:b]
        m = b - a

        occ = r[:, 0:8].copy().view(np.uint64).ravel()
        pcs = r[:, 8:24]

        bits = ((occ[:, None] >> bitpos) & np.uint64(1)).astype(np.uint8)
        rank = (np.cumsum(bits, axis=1, dtype=np.int16) - 1)

        lo = (pcs & 0x0F).astype(np.int16)
        hi = (pcs >> 4).astype(np.int16)
        codes = np.empty((m, 32), dtype=np.int16)
        codes[:, 0::2] = lo
        codes[:, 1::2] = hi

        piece_code = np.take_along_axis(codes, np.clip(rank, 0, 31).astype(np.intp), axis=1)
        feat = piece_code * 64 + sq64

        occupied = bits.astype(bool)
        counts = occupied.sum(axis=1)
        order = np.argsort(~occupied, axis=1, kind="stable")
        feat_sorted = np.take_along_axis(feat, order, axis=1)[:, :MAX_PIECES]
        keep = np.arange(MAX_PIECES)[None, :] < counts[:, None]
        idx[a:b] = np.where(keep, feat_sorted, PAD_INDEX).astype(np.int16)

    print(f"decodage en {time.time()-t_dec:.1f} s")

    if cache:
        np.savez(cache, idx=idx, score=score, stm=stm, result=result)
        print(f"cache ecrit : {cache}")

    return idx, score, stm, result


def mirror_table():
    """Index vu par les noirs, pour chaque index vu par les blancs."""
    t = np.empty(INPUT_SIZE + 1, dtype=np.int64)
    for i in range(INPUT_SIZE):
        code, sq = divmod(i, 64)
        t[i] = ((code + 6) % 12) * 64 + (sq ^ 56)
    t[PAD_INDEX] = PAD_INDEX
    return t


# --------------------------------------------------------------------------
# Le reseau
# --------------------------------------------------------------------------

class Perspective(nn.Module):
    def __init__(self, hidden, activation="crelu"):
        super().__init__()
        self.hidden = hidden
        self.activation = activation
        self.ft = nn.EmbeddingBag(INPUT_SIZE + 1, hidden, mode="sum", padding_idx=PAD_INDEX)
        self.ft_bias = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(2 * hidden, 1)

        nn.init.normal_(self.ft.weight, std=0.01)
        with torch.no_grad():
            self.ft.weight[PAD_INDEX].zero_()
            nn.init.normal_(self.out.weight, std=0.05)
            self.out.bias.zero_()

    def forward(self, idx_w, idx_b, stm):
        aw = self.ft(idx_w) + self.ft_bias
        ab = self.ft(idx_b) + self.ft_bias

        # stm = 0 -> blancs au trait
        s = stm.unsqueeze(1).float()
        us   = aw * (1 - s) + ab * s
        them = ab * (1 - s) + aw * s

        cu, ct = us.clamp(0, 1), them.clamp(0, 1)
        if self.activation == "screlu":
            # ReLU ecretee au carre : donne au reseau une non-linearite plus
            # marquee sans rien couter a l'inference, le carre se calculant
            # dans le meme produit vectoriel.
            cu, ct = cu * cu, ct * ct

        x = torch.cat([cu, ct], dim=1)
        return self.out(x).squeeze(1)

    def clip_weights(self):
        # |b| + 32|w| doit rester sous 32767/QA pour que l'accumulateur tienne
        # en int16 ; +/-1.98 laisse une marge confortable.
        with torch.no_grad():
            self.ft.weight.clamp_(-1.98, 1.98)
            self.ft_bias.clamp_(-1.98, 1.98)
            self.out.weight.clamp_(-1.98, 1.98)
            self.ft.weight[PAD_INDEX].zero_()


# --------------------------------------------------------------------------
# Export quantifie
# --------------------------------------------------------------------------

def export(model, path, hidden):
    with torch.no_grad():
        fw = model.ft.weight[:INPUT_SIZE].cpu().numpy()          # (768, HL)
        fb = model.ft_bias.cpu().numpy()                         # (HL,)
        ow = model.out.weight[0].cpu().numpy()                   # (2*HL,)
        ob = float(model.out.bias[0].cpu())

    fw_q = np.round(fw * QA).astype(np.int32)
    fb_q = np.round(fb * QA).astype(np.int32)
    ow_q = np.round(ow * QB).astype(np.int32)
    ob_q = int(round(ob * QA * QB))

    # Verification : l'accumulateur ne doit jamais deborder l'int16
    worst = np.abs(fb_q).max() + np.sort(np.abs(fw_q), axis=0)[-MAX_PIECES:].sum(axis=0).max()
    if worst >= 32767:
        raise SystemExit(f"debordement possible de l'accumulateur : {worst} >= 32767")
    for name, arr in (("entree", fw_q), ("biais", fb_q), ("sortie", ow_q)):
        if np.abs(arr).max() > 32767:
            raise SystemExit(f"poids {name} hors int16 : {np.abs(arr).max()}")

    with open(path, "wb") as f:
        f.write(struct.pack("<III", MAGIC, INPUT_SIZE, hidden))
        f.write(fw_q.astype("<i2").tobytes())     # rangee-majeur par caracteristique
        f.write(fb_q.astype("<i2").tobytes())
        f.write(ow_q.astype("<i2").tobytes())
        f.write(struct.pack("<i", ob_q))

    print(f"reseau ecrit : {path}  ({os.path.getsize(path)/1024:.0f} Ko)   "
          f"pire accumulateur {worst}/32767")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=r"C:\Onechess\data\nnue\*.bin")
    ap.add_argument("--cache", default=r"C:\Onechess\data\nnue_cache.npz")
    ap.add_argument("--out", default=r"C:\Onechess\engine\bin\onechess.nnue")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use", type=int, default=0,
                    help="ne garder que N positions apres chargement")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-eval", type=float, default=0.9,
                    help="poids du score moteur face au resultat de la partie")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--activation", default="crelu", choices=["crelu", "screlu"])
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"peripherique : {dev}"
          f"{' - ' + torch.cuda.get_device_name(0) if dev == 'cuda' else ''}")

    idx, score, stm, result = decode_shards(args.data, args.limit, args.cache)
    if args.use and args.use < len(score):
        idx, score, stm, result = (idx[:args.use], score[:args.use],
                                   stm[:args.use], result[:args.use])
    n = len(score)

    mirror = torch.from_numpy(mirror_table()).to(dev)

    perm = np.random.default_rng(args.seed).permutation(n)
    idx, score, stm, result = idx[perm], score[perm], stm[perm], result[perm]

    n_val = max(1024, int(n * args.val_frac))
    split = n - n_val

    t_idx    = torch.from_numpy(idx.astype(np.int64)).to(dev)
    t_score  = torch.from_numpy(score).to(dev)
    t_stm    = torch.from_numpy(stm).to(dev)
    t_result = torch.from_numpy(result).to(dev)

    # Score et resultat vus par le camp au trait
    sign = 1.0 - 2.0 * t_stm.float()
    t_score = t_score * sign
    t_result = torch.where(t_stm == 0, t_result, 1.0 - t_result)

    target = (args.lambda_eval * torch.sigmoid(t_score / SCALE)
              + (1.0 - args.lambda_eval) * t_result)

    model = Perspective(args.hidden, args.activation).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 30)

    print(f"{split:,} positions d'entrainement, {n_val:,} de validation, "
          f"HL={args.hidden}, lambda={args.lambda_eval}, activation={args.activation}")
    best = (1e9, 1e9, 0)

    steps = (split + args.batch - 1) // args.batch
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(split, device=dev)
        tot = 0.0

        for s in range(steps):
            b = order[s * args.batch:(s + 1) * args.batch]
            iw = t_idx[b]
            ib = mirror[iw]

            pred = model(iw, ib, t_stm[b])
            loss = ((torch.sigmoid(pred) - target[b]) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.clip_weights()
            tot += loss.item()

        sched.step()

        model.eval()
        with torch.no_grad():
            vb = torch.arange(split, n, device=dev)
            iw = t_idx[vb]
            pred = model(iw, mirror[iw], t_stm[vb])
            vloss = ((torch.sigmoid(pred) - target[vb]) ** 2).mean().item()
            # Erreur moyenne en centipions, plus parlante que la perte
            cp_err = (pred * SCALE - t_score[vb]).abs().mean().item()

        if vloss < best[0]:
            best = (vloss, cp_err, epoch)

        if not args.quiet:
            print(f"  epoque {epoch:3d}/{args.epochs}  perte {tot/steps:.5f}  "
                  f"val {vloss:.5f}  erreur {cp_err:6.1f} cp  "
                  f"({time.time()-t0:.0f} s)", flush=True)

    print(f"RESULTAT  activation={args.activation}  lambda={args.lambda_eval}  "
          f"HL={args.hidden}  epoques={args.epochs}  "
          f"meilleure_val={best[0]:.5f}  erreur={best[1]:.1f}cp  a_epoque={best[2]}  "
          f"val_finale={vloss:.5f}  erreur_finale={cp_err:.1f}cp", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    export(model, args.out, args.hidden)


if __name__ == "__main__":
    main()
