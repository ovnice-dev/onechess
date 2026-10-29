#!/usr/bin/env python3
"""Compte les positions reellement disponibles, SANS rien modifier.

Meme cle que la deduplication de l'entrainement (cles() de prep_lot.py,
identique a dedup_recs de train.py) : plateau + trait. Pour chaque groupe
de fichiers, et pour l'ensemble : positions brutes, positions distinctes,
et positions gardees a l'entrainement avec --dedup 4 (au plus 4 copies).

    python tools/compte_uniques.py
"""
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_lot import RECORD, cles

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPES = {
    "Lichess etiquete": ["data/nnue_elite/**/*.bin"],
    "autojeu (Kaggle + HF + PC)": ["data/autojeu/*.bin", "data/autojeu_pc/*.bin"],
}


def lire_cles(motifs):
    fs = sorted({f for m in motifs for f in glob.glob(os.path.join(RACINE, m), recursive=True)})
    n = sum(os.path.getsize(f) // RECORD for f in fs)
    k = np.empty(n, dtype=np.uint64)
    p = 0
    for f in fs:
        m = os.path.getsize(f) // RECORD
        mm = np.memmap(f, dtype=np.uint8, mode="r", shape=(m, RECORD))
        for a in range(0, m, 1 << 22):
            r = np.asarray(mm[a:a + (1 << 22)])
            k[p:p + len(r)] = cles(r)
            p += len(r)
        del mm
    assert p == n
    return k, len(fs)


def bilan(k, kmax=4, bloc=1 << 26):
    """k trie. Rend (distinctes, gardees avec au plus kmax copies)."""
    distinctes = gardees = 0
    prec = None
    rang_prec = -1
    for a in range(0, len(k), bloc):
        b = k[a:a + bloc]
        debut = np.empty(len(b), dtype=bool)
        debut[0] = prec is None or b[0] != prec
        np.not_equal(b[1:], b[:-1], out=debut[1:])
        idx = np.arange(len(b), dtype=np.int64)
        tete = np.where(debut, idx, -1)
        np.maximum.accumulate(tete, out=tete)
        rang = np.where(tete >= 0, idx - tete, idx + rang_prec + 1)
        distinctes += int(debut.sum())
        gardees += int((rang < kmax).sum())
        prec, rang_prec = b[-1], int(rang[-1])
    return distinctes, gardees


def main():
    t0 = time.time()
    tous = []
    for nom, motifs in GROUPES.items():
        k, nf = lire_cles(motifs)
        k.sort()
        d, g = bilan(k)
        print(f"{nom:28s} {nf:4d} fichiers  brutes {len(k)/1e6:7.1f} M  distinctes {d/1e6:7.1f} M  "
              f"gardees (<=4 copies) {g/1e6:7.1f} M   ({time.time() - t0:.0f} s)", flush=True)
        tous.append(k)
    k = np.concatenate(tous)
    del tous
    k.sort(kind="stable")
    d, g = bilan(k)
    print(f"{'ENSEMBLE':28s}                brutes {len(k)/1e6:7.1f} M  distinctes {d/1e6:7.1f} M  "
          f"gardees (<=4 copies) {g/1e6:7.1f} M   ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
