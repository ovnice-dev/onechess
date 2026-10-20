#!/usr/bin/env python3
"""Portrait d'un lot de positions etiquetees (.bin de 32 octets).

Sert a comparer un lot nouveau a un lot de reference AVANT de l'entrainer.
Le lot Lichess passait tous les controles de forme (signe, empaquetage,
aller-retour) et valait -464 Elo : ce qui compte, c'est la distribution. Cet
outil la decrit pour plusieurs lots cote a cote : scores, resultats, camp au
trait, nombre de pieces, doublons, et resultat moyen par tranche de score.

    python tools/check_bin.py data/nnue_big data/_tmp/elite_sample

Chaque lot est un repertoire de .bin ou un fichier. Au-dela de 4 M de
positions, on en prend un echantillon regulier.
"""

import glob
import os
import sys

import numpy as np

# Format ecrit par label.py : struct "<Q16shBBB3x". Les 16 octets de pieces
# sont lus comme deux entiers, ce qui suffit pour reperer les doublons.
REC = np.dtype([("occ", "<u8"), ("p1", "<u8"), ("p2", "<u8"), ("score", "<i2"),
                ("stm", "u1"), ("result", "u1"), ("r50", "u1"), ("pad", "V3")])
assert REC.itemsize == 32

MAX_PER_SET = 4_000_000
BINS = [(-10**9, -600), (-600, -300), (-300, -100), (-100, 100), (100, 300), (300, 600), (600, 10**9)]


def load(path):
    files = sorted(glob.glob(os.path.join(path, "*.bin"))) if os.path.isdir(path) else [path]
    sizes = [os.path.getsize(f) // 32 for f in files]
    total = sum(sizes)
    step = max(1, total // MAX_PER_SET)
    # Taille explicite : un fichier en cours d'ecriture peut finir sur un
    # enregistrement incomplet.
    parts = [np.asarray(np.memmap(f, dtype=REC, mode="r", shape=(n,))[::step])
             for f, n in zip(files, sizes) if n]
    return np.concatenate(parts), total


def popcount(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def pct(x):
    return f"{100.0 * x:5.1f} %"


def portrait(a):
    s = a["score"].astype(np.int64)            # point de vue des blancs
    stm = a["stm"].astype(np.int64)            # 0 = blancs au trait
    res = a["result"].astype(np.int64)         # 2 = blancs gagnent, 1 = nulle, 0 = noirs
    s_stm = np.where(stm == 0, s, -s)
    abs_s = np.abs(s_stm)
    pcs = popcount(a["occ"])

    out = {
        "score moyen (camp au trait)": f"{s_stm.mean():+.1f}",
        "ecart-type du score": f"{s_stm.std():.0f}",
        "|score| median": f"{np.median(abs_s):.0f}",
        "|score| < 50": pct((abs_s < 50).mean()),
        "|score| > 1000": pct((abs_s > 1000).mean()),
        "|score| > 2000": pct((abs_s > 2000).mean()),
        "gains blancs": pct((res == 2).mean()),
        "nulles": pct((res == 1).mean()),
        "gains noirs": pct((res == 0).mean()),
        "trait aux blancs": pct((stm == 0).mean()),
        "pieces (moyenne)": f"{pcs.mean():.1f}",
        "pieces 2-10": pct((pcs <= 10).mean()),
        "pieces 11-20": pct(((pcs > 10) & (pcs <= 20)).mean()),
        "pieces 21-32": pct((pcs > 20).mean()),
        "regle des 50 coups (moyenne)": f"{a['r50'].mean():.1f}",
    }
    dec = (res != 1) & (np.abs(s) > 200)
    out["score et vainqueur d'accord (|s|>200)"] = pct(((s > 0) == (res == 2))[dec].mean()) if dec.any() else "-"

    key = np.stack([a["occ"], a["p1"], a["p2"], a["stm"].astype(np.uint64)], axis=1)
    out["doublons dans l'echantillon"] = pct(1.0 - np.unique(key, axis=0).shape[0] / len(a))

    # Resultat moyen des blancs selon leur score : c'est la relation que le
    # reseau apprend quand on melange score et resultat.
    for lo, hi in BINS:
        m = (s >= lo) & (s < hi)
        name = f"score blancs [{max(lo, -9999):+d} ; {min(hi, 9999):+d}["
        out[name] = (f"{res[m].mean() / 2:.3f} ({pct(m.mean()).strip()})" if m.any() else "-")
    return out


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    cols = []
    for p in paths:
        a, total = load(p)
        d = portrait(a)
        d = {"positions": f"{total:,}", "echantillon": f"{len(a):,}", **d}
        cols.append((os.path.basename(os.path.normpath(p)), d))

    w0 = max(len(k) for k in cols[0][1])
    w = max(18, *(len(n) for n, _ in cols))
    print(f"{'':<{w0}}  " + "  ".join(f"{n:>{w}}" for n, _ in cols))
    for k in cols[0][1]:
        print(f"{k:<{w0}}  " + "  ".join(f"{d[k]:>{w}}" for _, d in cols))
    print("\nLes lignes 'score blancs [..[' donnent le resultat moyen des blancs "
          "(0 = perdu, 1 = gagne) et la part des positions dans la tranche.")


if __name__ == "__main__":
    main()
