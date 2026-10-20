"""Bilan d'un SPRT en lot : additionne les comptes pentanomiaux de tous les
fichiers .state trouves (tous morceaux, toutes vagues) et calcule Elo et LLR.

    python ci/bilan_lot.py --bornes 0 5 dossier [dossier...]

Derniere ligne : VERDICT H1 | VERDICT H0 | VERDICT CONTINUER (lue par sprt_lot.sh).
"""
import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from arena import elo_with_ci, llr_pentanomial  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bornes", nargs=2, type=float, default=[0.0, 5.0])
    ap.add_argument("dossiers", nargs="+")
    a = ap.parse_args()
    counts = [0] * 5
    w = d = l = 0
    fichiers = 0
    for dos in a.dossiers:
        for f in glob.glob(os.path.join(dos, "**", "*.state"), recursive=True):
            with open(f, encoding="utf-8") as fh:
                st = json.load(fh)
            counts = [x + y for x, y in zip(counts, st["counts"])]
            w += st["w"]; d += st["d"]; l += st["l"]
            fichiers += 1
    e0, e1 = a.bornes
    lo, hi = math.log(0.05 / 0.95), math.log(0.95 / 0.05)
    elo, ci = elo_with_ci(counts)
    llr = llr_pentanomial(counts, e0, e1)
    print(f"{fichiers} morceaux   {w + d + l} parties  +{w} ={d} -{l}")
    print(f"paires  0:{counts[0]}  0.5:{counts[1]}  1:{counts[2]}  1.5:{counts[3]}  2:{counts[4]}")
    print(f"Elo {elo:+.1f} +/- {ci:.1f}   LLR {llr:+.2f} [{lo:+.2f},{hi:+.2f}]  bornes [{e0:g} ; {e1:g}]")
    v = "H1" if llr >= hi else "H0" if llr <= lo else "CONTINUER"
    print(f"VERDICT {v}")


if __name__ == "__main__":
    main()
