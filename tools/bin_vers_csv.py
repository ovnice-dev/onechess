#!/usr/bin/env python3
"""Tire au hasard des positions d'un lot .bin et les rend en CSV "fen,result".

Sert au reetiquetage (essai 9.2) : on reprend des positions deja etiquetees
(Lichess, Stockfish 18 a 1 000 noeuds), on les repasse a label.py avec un
autre moteur, et on compare deux reseaux entraines sur les MEMES positions,
seule l'etiquette changeant.

Le format 32 octets ne garde ni les droits de roque ni la prise en passant :
le FEN produit n'en a pas ("-"). Le reseau ne les voit pas non plus ; seule la
recherche de l'etiqueteur peut en etre (tres rarement) affectee.

    python tools/bin_vers_csv.py --data "data/nnue_elite/**/*.bin" --n 20000000 \\
        --parts 5 --seed 1 --out data/reetiq/echantillon
      -> echantillon_1.bin ... _5.bin (enregistrements d'origine, intacts)
    python tools/bin_vers_csv.py --decode echantillon_1.bin --csv part_1.csv
"""

import argparse
import glob
import os

import numpy as np

RECORD = 32
CHARS = "PNBRQKpnbrqk"


def fen_of(r):
    """Plateau, trait et regle des 50 coups d'un enregistrement."""
    occ = int.from_bytes(bytes(r[0:8]), "little")
    board = [None] * 64
    k = 0
    for sq in range(64):
        if occ >> sq & 1:
            b = r[8 + (k >> 1)]
            code = (b >> 4) if (k & 1) else (b & 0x0F)
            board[sq] = CHARS[code]
            k += 1
    rows = []
    for rank in range(7, -1, -1):
        s, vide = "", 0
        for f in range(8):
            c = board[rank * 8 + f]
            if c is None:
                vide += 1
            else:
                if vide:
                    s += str(vide)
                    vide = 0
                s += c
        if vide:
            s += str(vide)
        rows.append(s)
    stm = "w" if r[26] == 0 else "b"
    return f"{'/'.join(rows)} {stm} - - {int(r[28])} 1"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", default=[])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--decode", default="", help="un .bin a convertir en CSV")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    if args.decode:
        r = np.fromfile(args.decode, dtype=np.uint8).reshape(-1, RECORD)
        res = {0: "0.0", 1: "0.5", 2: "1.0"}
        with open(args.csv, "w", encoding="ascii", newline="\n") as f:
            for x in r:
                f.write(f"{fen_of(x)},{res[int(x[27])]}\n")
        print(f"{len(r):,} positions -> {args.csv}")
        return

    files = sorted({f for m in args.data for f in glob.glob(m, recursive=True)})
    sizes = np.array([os.path.getsize(f) // RECORD for f in files])
    total = int(sizes.sum())
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(total, size=args.n, replace=False))
    starts = np.concatenate([[0], np.cumsum(sizes)])
    out = []
    for i, f in enumerate(files):
        a, b = np.searchsorted(pick, [starts[i], starts[i + 1]])
        if b > a:
            mm = np.memmap(f, dtype=np.uint8, mode="r", shape=(int(sizes[i]), RECORD))
            out.append(np.asarray(mm[pick[a:b] - starts[i]]))
    rec = np.concatenate(out)
    rec = rec[rng.permutation(len(rec))]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for p, part in enumerate(np.array_split(rec, args.parts), 1):
        part.tofile(f"{args.out}_{p}.bin")
    print(f"{len(rec):,} positions tirees parmi {total:,} ({len(files)} fichiers), "
          f"{args.parts} parties -> {args.out}_*.bin")


if __name__ == "__main__":
    main()
