#!/usr/bin/env python3
"""Extrait des positions de parties PGN vers le CSV "fen,result" de label.py.

Source visee : la Lichess Elite Database (database.nikonoel.fr), parties de
joueurs 2400+ contre 2200+ (2500+ contre 2300+ depuis decembre 2021), sans
bullet. C'est la meme NATURE de donnees que le CSV qui a donne les 16 M de
positions actuelles -- des positions de parties, pas des positions choisies
pour analyse. La lecon du 09-09 est que cette nature compte plus que le
volume : des evaluations Lichess pourtant justes, mais portant sur des
positions soumises a l'analyse, ont coute 40 cp.

Filtres appliques ici, les memes que dans label.py pour que les lots restent
melangeables : coup 8 au moins, pas d'echec. Le filtre de position tranquille,
lui, depend du meilleur coup trouve par Stockfish : label.py l'applique a
l'etiquetage.

Echantillonnage : au plus --per-game positions par partie, tirees au hasard.
Deux positions voisines d'une meme partie sont presque la meme position ; en
prendre trente ne vaut pas trente parties.

Les archives .zip sont lues sans etre extraites : decompresser les 65 mois sur
disque demanderait environ 40 Go. Chaque mois produit son propre CSV, ecrit
sous un nom provisoire puis renomme : une conversion interrompue ne laisse
jamais un fichier tronque qui passerait pour complet, et une relance reprend
aux mois manquants.
"""

import argparse
import glob
import io
import os
import random
import sys
import time
import zipfile
import zlib
from multiprocessing import Pool

import chess
import chess.pgn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import winjob

RESULT = {"1-0": "1.0", "0-1": "0.0", "1/2-1/2": "0.5"}


def open_pgn(path):
    # Le fichier commence par un BOM UTF-8 : sans utf-8-sig, la premiere
    # balise de la premiere partie serait illisible.
    if path.endswith(".zip"):
        z = zipfile.ZipFile(path)
        name = next(n for n in z.namelist() if n.endswith(".pgn"))
        return io.TextIOWrapper(z.open(name), encoding="utf-8-sig", errors="replace")
    return open(path, encoding="utf-8-sig", errors="replace")


def convert(task):
    path, out_dir, per_game, min_fullmove, seed = task
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, base + ".csv")
    tmp = out_path + ".part"

    # Graine stable par fichier : hash() de Python change a chaque processus.
    rng = random.Random(zlib.crc32(base.encode()) ^ seed)
    games = kept = 0
    t0 = time.time()

    with open_pgn(path) as f, open(tmp, "w", encoding="utf-8") as out:
        out.write("fen,result\n")
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            res = RESULT.get(game.headers.get("Result", "*"))
            if res is None:
                continue
            games += 1

            board = game.board()
            cands = []
            for move in game.mainline_moves():
                board.push(move)
                if board.fullmove_number >= min_fullmove and not board.is_check():
                    cands.append(board.fen())

            if len(cands) > per_game:
                cands = rng.sample(cands, per_game)
            for fen in cands:
                out.write(fen + "," + res + "\n")
            kept += len(cands)

    os.replace(tmp, out_path)
    return base, games, kept, time.time() - t0


def main():
    winjob.install()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn", default=r"C:\Onechess\data\lichess_elite\*.zip")
    ap.add_argument("--out", default=r"C:\Onechess\data\elite_csv")
    ap.add_argument("--per-game", type=int, default=12)
    ap.add_argument("--min-fullmove", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(args.pgn))

    def csv_of(p):
        return os.path.join(args.out, os.path.splitext(os.path.basename(p))[0] + ".csv")

    todo = [p for p in files if not os.path.exists(csv_of(p))]
    print(f"{len(files)} fichiers, {len(files) - len(todo)} deja convertis, {len(todo)} a faire   "
          f"{args.per_game} positions max par partie   {args.workers} travailleurs", flush=True)

    t0 = time.time()
    total_games = total_kept = 0
    tasks = [(p, args.out, args.per_game, args.min_fullmove, args.seed) for p in todo]
    with Pool(args.workers) as pool:
        for base, g, k, dt in pool.imap_unordered(convert, tasks):
            total_games += g
            total_kept += k
            print(f"  {base:26s} {g:9,} parties  {k:11,} positions  ({dt / 60:.1f} min)   "
                  f"total {total_kept:,}", flush=True)

    print(f"\ntermine : {total_games:,} parties, {total_kept:,} positions "
          f"en {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
