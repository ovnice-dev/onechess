#!/usr/bin/env python3
"""Mesure les positions en double dans le lot d'entrainement (etape D.4).

Une position est identifiee par son plateau (24 premiers octets de
l'enregistrement de 32) et le camp au trait (octet 26) ; le score et le
resultat ne comptent pas. Rien n'est modifie : on mesure d'abord.

    python tools/doublons.py "C:/Onechess/data/nnue_elite/*/*.bin"
"""
import argparse
import glob
import os
import sys

import numpy as np


def cles(path):
    r = np.fromfile(path, dtype=np.uint8)
    r = r[: len(r) // 32 * 32].reshape(-1, 32)
    q = r[:, :24].copy().view("<u8")                 # 3 mots de 64 bits
    k = q[:, 0] * np.uint64(0x9E3779B97F4A7C15)
    k ^= (q[:, 1] + np.uint64(0x632BE59BD9B4E019)) * np.uint64(0xC2B2AE3D27D4EB4F)
    k ^= (q[:, 2] + np.uint64(0x165667B19E3779F9)) * np.uint64(0x94D049BB133111EB)
    k ^= r[:, 26].astype(np.uint64) * np.uint64(0xD6E8FEB86659FD93)
    k ^= k >> np.uint64(31)
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motif")
    a = ap.parse_args()
    fichiers = sorted(glob.glob(a.motif))
    if not fichiers:
        sys.exit("aucun fichier")
    parties, source = [], []
    with np.errstate(over="ignore"):
        for i, f in enumerate(fichiers):
            k = cles(f)
            parties.append(k)
            source.append(np.full(len(k), i, dtype=np.int32))
    k = np.concatenate(parties)
    src = np.concatenate(source)
    n = len(k)
    _, premier, nb = np.unique(k, return_index=True, return_counts=True)
    u = len(nb)
    print(f"{len(fichiers)} fichiers, {n:,} positions, {u:,} distinctes : "
          f"{n - u:,} doublons ({100 * (n - u) / n:.2f} %)")
    print(f"positions vues 2 fois ou plus : {int((nb > 1).sum()):,} ; "
          f"la plus frequente : {int(nb.max()):,} fois")
    # Repartition par origine : Kaggle (kg_) ou PC
    kg = np.array(["kg_" in os.path.basename(f) for f in fichiers])
    garde = np.zeros(n, dtype=bool)
    garde[premier] = True
    for nom, m in (("PC", ~kg[src]), ("Kaggle", kg[src])):
        if m.any():
            print(f"  {nom:7s} {int(m.sum()):>12,} positions, "
                  f"{100 * (1 - garde[m].mean()):.2f} % en double d'une position deja vue")


if __name__ == "__main__":
    main()
