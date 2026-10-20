#!/usr/bin/env python3
"""Fabrique un jeu d'ouvertures equilibrees pour l'arene.

On tire des lignes dans un livre Polyglot, puis on ne garde que les positions
que Stockfish juge equilibrees a faible profondeur. Deux ouvertures desequilibrees
suffisent a fausser un match court : la moitie des paires devient decidee par
l'ouverture et non par les moteurs.
"""

import argparse
import os
import random
import sys

import chess
import chess.engine
import chess.polyglot


def sample_lines(book_path, count, plies, seed):
    rng = random.Random(seed)
    seen = set()
    fens = []

    with chess.polyglot.open_reader(book_path) as book:
        attempts = 0
        while len(fens) < count and attempts < count * 60:
            attempts += 1
            board = chess.Board()
            ok = True

            for _ in range(plies):
                entries = list(book.find_all(board))
                if not entries:
                    ok = False
                    break
                # Tirage uniforme parmi TOUS les coups du livre, et non
                # pondere : la ponderation ramene toujours a la meme poignee
                # de lignes principales, ce qui ne donne aucune diversite.
                entry = rng.choice(entries)
                if entry.move not in board.legal_moves:
                    ok = False
                    break
                board.push(entry.move)

            if not ok or board.is_game_over():
                continue

            key = board.board_fen() + (" w" if board.turn else " b")
            if key in seen:
                continue
            seen.add(key)
            fens.append(board.fen())

    return fens


def filter_balanced(fens, engine_path, depth, margin, threads):
    kept = []
    with chess.engine.SimpleEngine.popen_uci(engine_path) as eng:
        eng.configure({"Threads": threads, "Hash": 128})
        for i, fen in enumerate(fens, 1):
            board = chess.Board(fen)
            info = eng.analyse(board, chess.engine.Limit(depth=depth))
            sc = info["score"].white()
            if sc.is_mate():
                continue
            if abs(sc.score()) <= margin:
                kept.append(fen)
            if i % 200 == 0:
                print(f"  {i}/{len(fens)} evaluees, {len(kept)} gardees", file=sys.stderr,
                      flush=True)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="C:/Onechess/opening_book/komodo.bin")
    ap.add_argument("--out", default="C:/Onechess/data/openings.epd")
    ap.add_argument("--count", type=int, default=4000, help="lignes tirees avant filtrage")
    ap.add_argument("--plies", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--engine", default=None, help="Stockfish, pour le filtrage")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--margin", type=int, default=60, help="ecart maximal tolere, en cp")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    print(f"tirage de {args.count} lignes de {args.plies} demi-coups dans "
          f"{os.path.basename(args.book)}", file=sys.stderr)
    fens = sample_lines(args.book, args.count, args.plies, args.seed)
    print(f"{len(fens)} positions distinctes", file=sys.stderr)

    if args.engine:
        print(f"filtrage a profondeur {args.depth}, marge {args.margin} cp", file=sys.stderr)
        fens = filter_balanced(fens, args.engine, args.depth, args.margin, args.threads)
        print(f"{len(fens)} positions equilibrees", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for fen in fens:
            f.write(fen + "\n")

    print(f"ecrit : {args.out}  ({len(fens)} ouvertures)")


if __name__ == "__main__":
    main()
