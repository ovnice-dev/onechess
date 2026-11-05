"""Reordonne les neurones de la 1re couche d'un reseau a produit par paires
pour que les entrees nulles de l1 tombent par groupes de 4 (phase 10, A3).

Le calcul creux de l1 saute un groupe de 4 octets seulement s'ils sont tous
nuls. Sur R1024H, 16 % des octets sont non nuls mais 50 % des groupes : les
zeros sont eparpilles. On regroupe les neurones qui s'eteignent ensemble.
L'evaluation est exacte au bit pres (seul l'ordre des sommes entieres change).

  1. masques : ONE_NNZ_DUMP=masques.bin binaire compile avec -DONE_NNZ_STATS, bench
  2. python tools/ftperm.py reseau.nnue masques.bin sortie.nnue [--echantillons 200000]
"""
import argparse
import struct

import numpy as np

BASE_INPUTS, PAIR_INPUTS, THREAT_INPUTS = 768, 96 * 95 // 2, 768
PAIRS_FLAG, THREAT_FLAG, HIDDEN_FLAG, PAIRMUL_FLAG = 256, 512, 1024, 2048
MAGIC4 = 0x344E4E4F
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def permutation(act, hh):
    """act : (S, hh) booleens (neurone-produit non nul). Glouton : chaque groupe
    part du neurone le plus souvent nul, puis ajoute 3 fois celui qui garde le
    plus d'echantillons ou tout le groupe est nul."""
    nul = np.packbits(~act, axis=0)                       # (S/8, hh), bit = 1 si nul
    reste = list(np.argsort(act.sum(axis=0)))             # du plus souvent nul au moins
    ordre = []
    while reste:
        g = [reste.pop(0)]
        z = nul[:, g[0]].copy()
        for _ in range(3):
            cand = np.array(reste)
            score = POP[z[:, None] & nul[:, cand]].sum(axis=0)
            k = int(np.argmax(score))
            g.append(reste.pop(k))
            z &= nul[:, g[-1]]
        ordre += g
    return np.array(ordre)


def groupes_non_nuls(act, perm):
    a = act[:, perm].reshape(act.shape[0], -1, 4).any(axis=2)
    return 100.0 * a.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reseau")
    ap.add_argument("masques")
    ap.add_argument("sortie")
    ap.add_argument("--echantillons", type=int, default=200000)
    a = ap.parse_args()

    raw = bytearray(open(a.reseau, "rb").read())
    magic, hl, nb, mirror, ob_n = struct.unpack_from("<IIIII", raw, 0)
    act_flags = struct.unpack_from("<I", raw, 20)[0]
    if magic != MAGIC4 or not (act_flags & HIDDEN_FLAG) or not (act_flags & PAIRMUL_FLAG):
        raise SystemExit("reseau sans couches cachees a produit par paires : rien a reordonner")
    hh = hl // 2

    # Masques : une ligne par evaluation = hl bits (perspective du trait puis
    # l'autre) ; chaque perspective est un echantillon de hh neurones-produits.
    m = np.fromfile(a.masques, dtype=np.uint8).reshape(-1, hl // 8)
    bits = np.unpackbits(m, axis=1, bitorder="little").astype(bool)
    act = np.concatenate([bits[:, :hh], bits[:, hh:]], axis=0)
    rng = np.random.default_rng(0)
    idx = rng.permutation(act.shape[0])
    app, ctl = act[idx[:a.echantillons]], act[idx[a.echantillons:2 * a.echantillons]]
    perm = permutation(app, hh)
    ident = np.arange(hh)
    print(f"octets non nuls {100 * act.mean():.2f} %")
    print(f"groupes non nuls : avant {groupes_non_nuls(ctl, ident):.2f} %, "
          f"apres {groupes_non_nuls(ctl, perm):.2f} % (echantillons de controle)")

    # Ordre complet des neurones de la 1re couche : produit j <- ancien perm[j],
    # dans les deux moities (a[i] et a[i + hh] restent apparies).
    full = np.concatenate([perm, perm + hh])
    off = 24 + 64
    nfeat = nb * BASE_INPUTS
    def permute_cols(off, rows, dtype, size):
        n = rows * hl * size
        w = np.frombuffer(bytes(raw[off:off + n]), dtype=dtype).reshape(rows, hl)
        raw[off:off + n] = np.ascontiguousarray(w[:, full]).tobytes()
        return off + n
    off = permute_cols(off, nfeat, "<i2", 2)              # poids
    off = permute_cols(off, 1, "<i2", 2)                  # biais
    if act_flags & PAIRS_FLAG:
        off = permute_cols(off, PAIR_INPUTS, "<i2", 2)
    if act_flags & THREAT_FLAG:
        off = permute_cols(off, THREAT_INPUTS, "<i2", 2)
    l1, l2 = struct.unpack_from("<II", raw, off)
    off += 8
    # l1 : (ob_n, l1, hl) int8, entrees = [produits du trait | produits de l'autre]
    n = ob_n * l1 * hl
    w1 = np.frombuffer(bytes(raw[off:off + n]), dtype="i1").reshape(ob_n, l1, hl)
    raw[off:off + n] = np.ascontiguousarray(w1[:, :, full]).tobytes()
    open(a.sortie, "wb").write(raw)
    print("ecrit", a.sortie)


if __name__ == "__main__":
    main()
