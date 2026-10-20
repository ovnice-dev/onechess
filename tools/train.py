#!/usr/bin/env python3
"""Entrainement du reseau NNUE de Onechess.

Architecture : (768 x buckets -> HL) x2 perspectives -> 1.

Les 768 entrees de base sont 12 types de pieces x 64 cases, vues depuis le
camp considere. Le decoupage en buckets les duplique selon la case du roi du
camp considere : c'est ce qui donne au reseau la position d'une piece
RELATIVEMENT au roi, information qu'un jeu de 768 entrees ne peut pas
exprimer -- et c'est pour cela qu'un tel jeu plafonne quelle que soit la
quantite de donnees ou la taille du reseau.

Le miroir horizontal replie l'echiquier sur sa moitie dame quand le roi est
sur les colonnes e-h : deux fois plus d'exemples par poids, gratuitement.

Format de sortie, auto-descriptif pour que le moteur n'ait pas a etre
recompile quand le decoupage change :

    uint32 magic "ONN2", uint32 hidden, uint32 nBuckets, uint32 mirror,
    uint8  bucketLayout[64],
    int16  ftWeights[nBuckets*768][hidden], int16 ftBias[hidden],
    int16  outWeights[2*hidden], int32 outBias
"""

import argparse
import glob
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (BUCKET_LAYOUTS, bucket_count, compact_layout,   # noqa: E402
                      output_bucket)

BASE_INPUTS = 768
# Paires de pions (PP_3Wide de Stockfish 19) : toute paire de pions, de
# n'importe quelles couleurs, sur la meme colonne ou deux colonnes voisines.
# 96 identifiants de pion (couleur relative x 48 cases des rangees 2 a 7),
# paires non ordonnees : 96 * 95 / 2 entrees.
PAIR_INPUTS = 96 * 95 // 2
MAX_PAIRS   = 64      # borne pour le calcul du pire accumulateur
PAIRS_FLAG  = 256     # bit du champ « activation » de l'en-tete .nnue
MAX_PIECES  = 32
QA, QB      = 255, 64
SCALE       = 400
MAGIC       = 0x344E4E4F        # "ONN4" : en-tete avec activation
ACTIVATIONS = {"crelu": 0, "screlu": 1}


# --------------------------------------------------------------------------
# Lecture des enregistrements binaires
# --------------------------------------------------------------------------

def decode_shards(pattern, limit=0, cache=None):
    """Rend (codes, sqs, score, stm, result).

    codes[i, k] = code piece 0..11 (12 = vide), sqs[i, k] = case 0..63.
    On garde la description brute plutot que des index deja calcules : cela
    permet de changer de jeu de caracteristiques sans relire les donnees.
    """
    if cache and os.path.exists(cache):
        z = np.load(cache)
        print(f"cache lu : {cache}  ({len(z['score']):,} positions)")
        return z["codes"], z["sqs"], z["score"], z["stm"], z["result"]

    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"aucun fichier ne correspond a {pattern}")

    chunks, total = [], 0
    for fn in files:
        raw = np.fromfile(fn, dtype=np.uint8)
        n = len(raw) // 32
        if n:
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
    result = recs[:, 27].astype(np.float32) / 2.0

    codes = np.full((n, MAX_PIECES), 12, dtype=np.uint8)
    sqs   = np.zeros((n, MAX_PIECES), dtype=np.uint8)

    bitpos = np.arange(64, dtype=np.uint64)
    sq64   = np.arange(64, dtype=np.uint8)

    CHUNK = 250_000
    t0 = time.time()
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        r = recs[a:b]
        m = b - a

        occ = r[:, 0:8].copy().view(np.uint64).ravel()
        pcs = r[:, 8:24]

        bits = ((occ[:, None] >> bitpos) & np.uint64(1)).astype(np.uint8)
        rank = np.cumsum(bits, axis=1, dtype=np.int16) - 1

        nib = np.empty((m, 32), dtype=np.uint8)
        nib[:, 0::2] = pcs & 0x0F
        nib[:, 1::2] = pcs >> 4

        code_at = np.take_along_axis(nib, np.clip(rank, 0, 31).astype(np.intp), axis=1)

        occupied = bits.astype(bool)
        counts   = occupied.sum(axis=1)
        order    = np.argsort(~occupied, axis=1, kind="stable")

        codes[a:b] = np.where(
            np.arange(MAX_PIECES)[None, :] < counts[:, None],
            np.take_along_axis(code_at, order, axis=1)[:, :MAX_PIECES], 12)
        sqs[a:b] = np.take_along_axis(np.broadcast_to(sq64, (m, 64)), order,
                                      axis=1)[:, :MAX_PIECES]

    print(f"decodage en {time.time() - t0:.1f} s")

    if cache:
        np.savez(cache, codes=codes, sqs=sqs, score=score, stm=stm, result=result)
        print(f"cache ecrit : {cache}")

    return codes, sqs, score, stm, result


def dedup_recs(recs, k):
    """Garde au plus k exemplaires de chaque position (plateau + trait).

    Le lot Elite contient 11,4 % de doublons, surtout des positions
    d'ouverture (jusqu'a 7 151 fois la meme) : sans plafond, l'entrainement
    leur donne un poids demesure. On garde les k premiers exemplaires.
    """
    with np.errstate(over="ignore"):
        q = recs[:, :24].copy().view("<u8")
        key = q[:, 0] * np.uint64(0x9E3779B97F4A7C15)
        key ^= (q[:, 1] + np.uint64(0x632BE59BD9B4E019)) * np.uint64(0xC2B2AE3D27D4EB4F)
        key ^= (q[:, 2] + np.uint64(0x165667B19E3779F9)) * np.uint64(0x94D049BB133111EB)
        key ^= recs[:, 26].astype(np.uint64) * np.uint64(0xD6E8FEB86659FD93)
    order = np.argsort(key, kind="stable")
    ks    = key[order]
    debut = np.r_[True, ks[1:] != ks[:-1]]
    grp   = np.cumsum(debut) - 1
    rang  = np.arange(len(ks)) - np.flatnonzero(debut)[grp]
    garde = np.zeros(len(recs), dtype=bool)
    garde[order[rang < k]] = True
    print(f"doublons : {len(recs) - int(garde.sum()):,} positions retirees "
          f"(au plus {k} exemplaires par position), {int(garde.sum()):,} gardees")
    return recs[garde]


def read_raw(pattern, limit=0):
    """Rend les enregistrements bruts (n, 32) sans rien decoder."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"aucun fichier ne correspond a {pattern}")

    chunks, total = [], 0
    for fn in files:
        raw = np.fromfile(fn, dtype=np.uint8)
        k = len(raw) // 32
        if k:
            chunks.append(raw[: k * 32].reshape(k, 32))
            total += k
        if limit and total >= limit:
            break

    recs = np.concatenate(chunks, axis=0)
    if limit:
        recs = recs[:limit]
    print(f"{len(recs):,} positions lues dans {len(chunks)} fichiers (brut, 32 o)")
    return recs


class PreDecoded:
    """Tout est decode a l'avance et reside sur le GPU : 92 octets par
    position. Mur a environ 55 M de positions sur une carte de 6,4 Go."""

    def __init__(self, codes, sqs, stm, obuck, score, target):
        self.c, self.s   = codes, sqs
        self.stm, self.ob = stm, obuck
        self.sc, self.tg = score, target

    def batch(self, sel):
        return (self.c[sel], self.s[sel], self.stm[sel], self.ob[sel],
                self.sc[sel], self.tg[sel])


class RawData:
    """L'enregistrement brut de 32 octets reste sur le GPU, le decodage se
    fait par lot.

    32 octets par position au lieu de 92 : le plafond passe d'environ 55 M
    a 190 M de positions sur les 6,4 Go de la carte. Le decodage est un
    cumul de bits sur 64 elements par position -- negligeable devant la
    passe avant/arriere -- et il supprime le fichier de cache .npz.
    """

    def __init__(self, recs, device, out_buckets, lambda_eval):
        self.t    = torch.from_numpy(recs).to(device)
        self.ob_n = out_buckets
        self.lam  = lambda_eval
        self.bit8 = torch.arange(8, device=device)
        self.k32  = torch.arange(32, device=device)

    def batch(self, sel):
        r = self.t[sel].long()
        B = r.shape[0]

        # Les 8 premiers octets sont le bitboard d'occupation, petit-boutiste :
        # l'octet j porte les cases 8j a 8j+7.
        bits   = ((r[:, 0:8].unsqueeze(2) >> self.bit8) & 1).reshape(B, 64)
        counts = bits.sum(1, keepdim=True)

        # Les 16 suivants sont 32 quartets : le k-ieme decrit la k-ieme case
        # occupee, dans l'ordre croissant des cases.
        pcs = r[:, 8:24]
        nib = torch.empty(B, 32, dtype=torch.long, device=r.device)
        nib[:, 0::2] = pcs & 0x0F
        nib[:, 1::2] = pcs >> 4

        codes = torch.where(self.k32 < counts, nib, 12)
        sqs   = torch.argsort(bits, dim=1, descending=True, stable=True)[:, :32]

        sc  = r[:, 24] | (r[:, 25] << 8)
        sc  = torch.where(sc >= 32768, sc - 65536, sc).float()   # int16 signe
        stm = r[:, 26]
        res = r[:, 27].float() / 2.0

        sign = 1.0 - 2.0 * stm.float()
        sc   = sc * sign
        res  = torch.where(stm == 0, res, 1.0 - res)
        tg   = self.lam * torch.sigmoid(sc / SCALE) + (1.0 - self.lam) * res

        ob = torch.clamp(output_bucket(counts.squeeze(1), self.ob_n),
                         max=self.ob_n - 1)
        return codes, sqs, stm, ob, sc, tg


# --------------------------------------------------------------------------
# Index de caracteristiques, calcules sur GPU par lot
# --------------------------------------------------------------------------

class FeatureMap:
    def __init__(self, bucket_name, mirror, device):
        self.mirror   = mirror
        self.name     = bucket_name
        # "mat4" : melange d'experts a la Triumviratus 8.0. Le bucket d'entree
        # ne depend pas du roi mais du materiel, avec la meme formule que les
        # buckets de sortie. Mesure d'erreur seulement : le moteur ne sait pas
        # encore lire ce format, d'ou un layout de roi nul a l'export.
        self.material = bucket_name == "mat4"
        if self.material:
            self.nbuckets = 4
            self.layout   = torch.zeros(64, dtype=torch.long, device=device)
        else:
            self.nbuckets = bucket_count(bucket_name, mirror)
            self.layout   = torch.tensor(compact_layout(bucket_name, mirror),
                                         dtype=torch.long, device=device)
        self.n_inputs = self.nbuckets * BASE_INPUTS
        self.pad      = self.n_inputs

    def indices(self, codes, sqs, persp):
        """codes/sqs : (B, 32) int64. persp : 0 = blancs, 1 = noirs."""
        valid  = codes < 12
        colour = torch.div(codes, 6, rounding_mode="floor")
        ptype  = codes % 6

        pidx = ((colour != persp).long() * 6 + ptype)
        rel  = sqs ^ (56 * persp)

        # Case du roi de la perspective, dans le meme repere
        is_king = (codes == (5 + 6 * persp)).long()
        ksq = (rel * is_king).sum(dim=1, keepdim=True)

        if self.mirror:
            flip = (((ksq & 7) >= 4).long()) * 7
            rel  = rel ^ flip
            ksq  = ksq ^ flip

        base = pidx * 64 + rel                       # sans le bucket
        if self.material:
            n    = valid.sum(dim=1, keepdim=True)
            feat = torch.clamp((n - 2) // 8, 0, 3) * BASE_INPUTS + base
        else:
            feat = self.layout[ksq] * BASE_INPUTS + base

        pad_f = torch.full_like(feat, self.pad)
        pad_v = torch.full_like(base, BASE_INPUTS)
        return (torch.where(valid, feat, pad_f),
                torch.where(valid, base, pad_v))


def pair_indices(codes, sqs, persp):
    """Paires de pions vues d'une perspective. codes/sqs : (B, 32) int64.

    Rend (B, K) : les indices des paires presentes, completes par PAIR_INPUTS
    (ligne nulle). K est le plus grand nombre de paires du lot : on ne
    rassemble que ce qui existe.
    """
    pawn   = (codes < 12) & (codes % 6 == 0)
    colour = torch.div(codes, 6, rounding_mode="floor")
    rel    = sqs ^ (56 * persp)
    # int32 : les tenseurs (B, 32, 32) restent quatre fois plus petits
    pid    = (48 * (colour != persp).long() + (rel - 8).clamp(0, 47)).to(torch.int32)
    fil    = (rel & 7).to(torch.int32)

    a_ok = pawn.unsqueeze(2) & pawn.unsqueeze(1)
    tri  = torch.triu(torch.ones(codes.shape[1], codes.shape[1], dtype=torch.bool,
                                 device=codes.device), diagonal=1)
    near = (fil.unsqueeze(2) - fil.unsqueeze(1)).abs() <= 1
    ok   = a_ok & near & tri

    hi  = torch.maximum(pid.unsqueeze(2), pid.unsqueeze(1))
    lo  = torch.minimum(pid.unsqueeze(2), pid.unsqueeze(1))
    idx = torch.where(ok, hi * (hi - 1) // 2 + lo, PAIR_INPUTS).flatten(1)

    k = int(ok.flatten(1).sum(dim=1).max().item()) if idx.numel() else 0
    idx, _ = torch.sort(idx, dim=1)
    return idx[:, :max(k, 1)].long()


# --------------------------------------------------------------------------

class Perspective(nn.Module):
    def __init__(self, n_inputs, hidden, out_buckets=1, factorizer=False,
                 activation="crelu", pairs=False):
        super().__init__()
        self.hidden      = hidden
        self.pairs       = pairs
        if pairs:
            self.pp = nn.EmbeddingBag(PAIR_INPUTS + 1, hidden, mode="sum",
                                      padding_idx=PAIR_INPUTS)
            nn.init.normal_(self.pp.weight, std=0.005)
            with torch.no_grad():
                self.pp.weight[PAIR_INPUTS].zero_()
        self.pad         = n_inputs
        self.out_buckets = out_buckets
        self.activation  = activation
        self.factorizer  = factorizer and n_inputs > BASE_INPUTS
        self.ft      = nn.EmbeddingBag(n_inputs + 1, hidden, mode="sum",
                                       padding_idx=n_inputs)

        # Tronc commun : les memes 768 caracteristiques, sans distinction de
        # bucket. Chaque bucket n'a plus qu'a apprendre son ECART au tronc, et
        # le tronc, lui, profite de toutes les positions. C'est ce qui rend les
        # buckets de roi apprenables sans des centaines de millions d'exemples.
        # A l'export les deux tables sont repliees en une seule : le moteur
        # n'en sait rien.
        if self.factorizer:
            self.ft_v = nn.EmbeddingBag(BASE_INPUTS + 1, hidden, mode="sum",
                                        padding_idx=BASE_INPUTS)
            nn.init.normal_(self.ft_v.weight, std=0.01)
            with torch.no_grad():
                self.ft_v.weight[BASE_INPUTS].zero_()
        self.ft_bias = nn.Parameter(torch.zeros(hidden))
        self.out     = nn.Linear(2 * hidden, out_buckets)

        nn.init.normal_(self.ft.weight, std=0.01)
        with torch.no_grad():
            self.ft.weight[n_inputs].zero_()
            nn.init.normal_(self.out.weight, std=0.05)
            self.out.bias.zero_()

    def forward(self, idx_w, idx_b, stm, obuck):
        aw = self.ft(idx_w[0]) + self.ft_bias
        ab = self.ft(idx_b[0]) + self.ft_bias
        if self.factorizer:
            aw = aw + self.ft_v(idx_w[1])
            ab = ab + self.ft_v(idx_b[1])
        if self.pairs:
            aw = aw + self.pp(idx_w[2])
            ab = ab + self.pp(idx_b[2])

        s = stm.unsqueeze(1).float()          # 0 = blancs au trait
        us   = aw * (1 - s) + ab * s
        them = ab * (1 - s) + aw * s

        us, them = us.clamp(0, 1), them.clamp(0, 1)
        if self.activation == "screlu":
            us, them = us * us, them * them
        y = self.out(torch.cat([us, them], dim=1))
        if self.out_buckets == 1:
            return y.squeeze(1)
        return y.gather(1, obuck.unsqueeze(1)).squeeze(1)

    def clip_weights(self):
        with torch.no_grad():
            self.ft.weight.clamp_(-1.98, 1.98)
            self.ft_bias.clamp_(-1.98, 1.98)
            self.out.weight.clamp_(-1.98, 1.98)
            self.ft.weight[self.pad].zero_()
            if self.factorizer:
                self.ft_v.weight.clamp_(-1.98, 1.98)
                self.ft_v.weight[BASE_INPUTS].zero_()
            if self.pairs:
                self.pp.weight.clamp_(-1.98, 1.98)
                self.pp.weight[PAIR_INPUTS].zero_()


def export(model, fmap, path, hidden):
    with torch.no_grad():
        fw = model.ft.weight[:fmap.n_inputs].cpu().numpy()
        if model.factorizer:
            # Le tronc commun est additionne a chaque bucket : le fichier
            # produit est un reseau ordinaire, sans trace de la factorisation.
            shared = model.ft_v.weight[:BASE_INPUTS].cpu().numpy()
            fw = (fw.reshape(fmap.nbuckets, BASE_INPUTS, -1)
                  + shared[None, :, :]).reshape(fmap.n_inputs, -1)
        fb = model.ft_bias.cpu().numpy()
        pw = model.pp.weight[:PAIR_INPUTS].cpu().numpy() if model.pairs else None
        ow = model.out.weight.cpu().numpy()          # (outBuckets, 2*hidden)
        ob = model.out.bias.cpu().numpy()            # (outBuckets,)

    fw_q = np.round(fw * QA).astype(np.int32)
    fb_q = np.round(fb * QA).astype(np.int32)
    ow_q = np.round(ow * QB).astype(np.int32)
    ob_q = np.round(ob * QA * QB).astype(np.int32)

    # Pire accumulateur possible : biais + les 32 plus gros poids d'une colonne
    pw_q  = np.round(pw * QA).astype(np.int32) if pw is not None else None
    worst = np.abs(fb_q).max() + np.sort(np.abs(fw_q), axis=0)[-MAX_PIECES:].sum(axis=0).max()
    if pw_q is not None:
        worst += np.sort(np.abs(pw_q), axis=0)[-MAX_PAIRS:].sum(axis=0).max()
    if worst >= 32767:
        raise SystemExit(f"debordement de l'accumulateur : {worst} >= 32767")
    for nom, arr in (("entree", fw_q), ("biais", fb_q), ("sortie", ow_q)):
        if np.abs(arr).max() > 32767:
            raise SystemExit(f"poids {nom} hors int16 : {np.abs(arr).max()}")

    layout = [0] * 64 if fmap.material else compact_layout(fmap.name, fmap.mirror)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIIIII", MAGIC, hidden, fmap.nbuckets,
                            2 if fmap.material else int(fmap.mirror), model.out_buckets,
                            ACTIVATIONS[model.activation] | (PAIRS_FLAG if model.pairs else 0)))
        f.write(bytes(layout))
        f.write(fw_q.astype("<i2").tobytes())
        f.write(fb_q.astype("<i2").tobytes())
        if pw_q is not None:
            f.write(pw_q.astype("<i2").tobytes())   # PAIR_INPUTS x hidden, apres le biais
        f.write(ow_q.astype("<i2").tobytes())       # rangee-majeur par bucket
        f.write(ob_q.astype("<i4").tobytes())

    print(f"reseau ecrit : {path}  ({os.path.getsize(path)/1e6:.1f} Mo)   "
          f"roi={fmap.nbuckets} miroir={int(fmap.mirror)} "
          f"sortie={model.out_buckets} {model.activation}   pire accumulateur {worst}/32767")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=r"C:\Onechess\data\nnue\*.bin")
    ap.add_argument("--cache", default=r"C:\Onechess\data\nnue_cache2.npz")
    ap.add_argument("--out", default=r"C:\Onechess\engine\bin\onechess.nnue")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--buckets", default="plain", choices=list(BUCKET_LAYOUTS) + ["mat4"])
    ap.add_argument("--mirror", type=int, default=0)
    ap.add_argument("--out-buckets", type=int, default=1, choices=[1, 2, 4, 8])
    ap.add_argument("--activation", default="crelu",
                    choices=list(ACTIVATIONS),
                    help="crelu = clamp(a,0,1), screlu = clamp(a,0,1)^2")
    ap.add_argument("--factorizer", type=int, default=1,
                    help="tronc commun partage entre buckets de roi")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-eval", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--val-chunk", type=int, default=65536,
                    help="taille des morceaux de validation (le resultat ne "
                         "doit pas en dependre)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-decode", type=int, default=1,
                    help="garder les enregistrements bruts sur le GPU et decoder "
                         "par lot : 32 o/position au lieu de 92")
    ap.add_argument("--init", default="",
                    help="poids flottants (.pt) a reprendre : affinage d'un reseau deja entraine")
    ap.add_argument("--save-pt", default="",
                    help="sauver aussi les poids flottants de la meilleure epoque, pour un affinage ulterieur")
    ap.add_argument("--dedup", type=int, default=0,
                    help="garde au plus N exemplaires d'une meme position (0 = tout)")
    ap.add_argument("--pairs", type=int, default=0,
                    help="1 : entrees paires de pions (PP_3Wide de Stockfish 19)")
    ap.add_argument("--checkpoint", default="",
                    help="point de reprise ecrit a chaque epoque ; s'il existe au "
                         "lancement, l'entrainement reprend a l'epoque suivante")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.gpu_decode:
        recs = read_raw(args.data, args.limit)
        if args.dedup:
            recs = dedup_recs(recs, args.dedup)
        if args.use and args.use < len(recs):
            recs = recs[:args.use]
        n    = len(recs)
        recs = recs[np.random.default_rng(args.seed).permutation(n)]
        data = RawData(recs, dev, args.out_buckets, args.lambda_eval)
        print(f"decodage par lot : {n * 32 / 1e9:.2f} Go sur la carte")
    else:
        codes, sqs, score, stm, result = decode_shards(args.data, args.limit, args.cache)
        if args.use and args.use < len(score):
            codes, sqs, score, stm, result = (codes[:args.use], sqs[:args.use],
                                              score[:args.use], stm[:args.use],
                                              result[:args.use])
        n = len(score)

        perm = np.random.default_rng(args.seed).permutation(n)
        codes, sqs, score, stm, result = (codes[perm], sqs[perm], score[perm],
                                          stm[perm], result[perm])

        # Le bucket de sortie ne depend que du materiel restant
        piece_count = (codes < 12).sum(axis=1).astype(np.int64)
        obuck = np.minimum(output_bucket(piece_count, args.out_buckets),
                           args.out_buckets - 1)

        # Conserves en uint8 sur le GPU : en int64 il faudrait 11 Go pour 22 M
        # positions. La conversion se fait par lot, ou elle ne coute rien.
        t_codes  = torch.from_numpy(codes).to(dev)
        t_sqs    = torch.from_numpy(sqs).to(dev)
        t_score  = torch.from_numpy(score).to(dev)
        t_stm    = torch.from_numpy(stm).to(dev)
        t_result = torch.from_numpy(result).to(dev)
        t_obuck  = torch.from_numpy(obuck).to(dev)

        sign     = 1.0 - 2.0 * t_stm.float()
        t_score  = t_score * sign
        t_result = torch.where(t_stm == 0, t_result, 1.0 - t_result)
        target   = (args.lambda_eval * torch.sigmoid(t_score / SCALE)
                    + (1.0 - args.lambda_eval) * t_result)
        data = PreDecoded(t_codes, t_sqs, t_stm, t_obuck, t_score, target)

    n_val = max(2048, int(n * args.val_frac))
    split = n - n_val

    fmap  = FeatureMap(args.buckets, bool(args.mirror), dev)
    model = Perspective(fmap.n_inputs, args.hidden, args.out_buckets,
                        bool(args.factorizer), args.activation, bool(args.pairs)).to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev))
        print(f"reprise des poids de {args.init}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr / 30)

    print(f"{dev} | {split:,} entrainement / {n_val:,} validation | HL={args.hidden} "
          f"| roi={args.buckets}({fmap.nbuckets}) miroir={args.mirror} "
          f"sortie={args.out_buckets} tronc={int(model.factorizer)} "
          f"| {fmap.n_inputs:,} entrees{' + paires de pions' if args.pairs else ''}, "
          f"{(fmap.n_inputs + PAIR_INPUTS * args.pairs)*args.hidden/1e6:.1f} M poids")

    steps = (split + args.batch - 1) // args.batch
    best       = (1e9, 1e9, 0)
    best_state = None
    t0         = time.time()

    def feats(c, s):
        c, s = c.long(), s.long()
        if args.pairs:
            return (fmap.indices(c, s, 0) + (pair_indices(c, s, 0),),
                    fmap.indices(c, s, 1) + (pair_indices(c, s, 1),))
        return fmap.indices(c, s, 0), fmap.indices(c, s, 1)

    def eval_set(sel, chunk=args.val_chunk):
        """Perte et erreur en centipions sur un ensemble, par morceaux.

        Un passage d'un seul tenant sur les 320 000 positions de validation
        demande plusieurs centaines de Mo par tenseur intermediaire : la carte
        est deja a 5,9 Go sur 6,1 avec HL=512, et deborderait a 1024.

        Les sommes sont accumulees en Python, donc en double precision : le
        resultat est au moins aussi exact qu'une moyenne unique en float32.
        """
        s_loss = s_err = 0.0
        for a in range(0, len(sel), chunk):
            sub = sel[a:a + chunk]
            c, sq, stm_b, ob_b, sc_b, tg_b = data.batch(sub)
            iw, ib = feats(c, sq)
            pred   = model(iw, ib, stm_b, ob_b)
            s_loss += ((torch.sigmoid(pred) - tg_b) ** 2).sum().item()
            s_err  += (pred * SCALE - sc_b).abs().sum().item()
        return s_loss / len(sel), s_err / len(sel)

    cp_err = float("nan")
    first  = 1
    # Reprise : un arret (veille, PC rendu a l'utilisateur) ne coute plus que
    # l'epoque en cours. Les donnees et la separation validation/entrainement
    # dependent de --seed, donc elles sont les memes au redemarrage.
    if args.checkpoint and os.path.exists(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        best, best_state = tuple(ck["best"]), ck["best_state"]
        torch.set_rng_state(ck["rng"].cpu())
        if dev == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        first = ck["epoch"] + 1
        print(f"reprise du point {args.checkpoint} : epoque {first}/{args.epochs}, "
              f"meilleure {best[1]:.1f} cp", flush=True)
    for epoch in range(first, args.epochs + 1):
        model.train()
        order = torch.randperm(split, device=dev)
        tot = 0.0

        for k in range(steps):
            sel = order[k * args.batch:(k + 1) * args.batch]
            c, sq, stm_b, ob_b, _, tg_b = data.batch(sel)
            iw, ib = feats(c, sq)
            loss = ((torch.sigmoid(model(iw, ib, stm_b, ob_b)) - tg_b) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.clip_weights()
            tot += loss.item()

        sched.step()

        model.eval()
        with torch.no_grad():
            vsel = torch.arange(split, n, device=dev)

            # Erreur sur un echantillon des donnees D'ENTRAINEMENT : si elle
            # rejoint celle de validation, le reseau ne peut pas representer
            # davantage ; si elle est bien plus basse, c'est la generalisation
            # qui bloque, donc les donnees.
            tsel = torch.arange(0, min(n_val, split), device=dev)

            vloss, cp_err = eval_set(vsel)
            _,     tr_err = eval_set(tsel)

        if cp_err < best[1]:
            best = (vloss, cp_err, epoch)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if args.checkpoint:
            tmp = args.checkpoint + ".tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch, "best": list(best),
                        "best_state": best_state, "rng": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state() if dev == "cuda" else None},
                       tmp)
            os.replace(tmp, args.checkpoint)

        if not args.quiet:
            print(f"  epoque {epoch:3d}/{args.epochs}  perte {tot/steps:.5f}  "
                  f"val {vloss:.5f}  erreur {cp_err:6.1f} cp  "
                  f"({time.time()-t0:.0f} s)", flush=True)

    print(f"RESULTAT  roi={args.buckets} miroir={args.mirror} "
          f"sortie={args.out_buckets} tronc={int(model.factorizer)} HL={args.hidden} "
          f"n={n} epoques={args.epochs}  meilleure_erreur={best[1]:.1f}cp "
          f"(epoque {best[2]})  finale={cp_err:.1f}cp  "
          f"ENTRAINEMENT={tr_err:.1f}cp", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"export de l'epoque {best[2]} ({best[1]:.1f} cp), et non de la derniere")

    export(model, fmap, args.out, args.hidden)
    if args.save_pt:
        torch.save(model.state_dict(), args.save_pt)
        print(f"poids flottants ecrits : {args.save_pt}")


if __name__ == "__main__":
    main()
