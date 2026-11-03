#!/usr/bin/env python3
"""Positions aleatoires a 5-6 pieces (tables 3-4 pieces chargees) : on compte
les scores affiches qui ne sont ni une evaluation (|cp| < 13000), ni un gain
des tables (|cp| dans 19754..20000), ni un mat."""
import os
import random
import subprocess
import sys

import chess

engine, n = sys.argv[1], int(sys.argv[2])
rng = random.Random(21)
TB = os.environ["SYZYGY"]


def pos():
    while True:
        b = chess.Board(None)
        sq = rng.sample(range(64), 6)
        b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sq[1], chess.Piece(chess.KING, chess.BLACK))
        pool = [chess.QUEEN, chess.ROOK, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN, chess.PAWN]
        for s in sq[2:2 + rng.choice([3, 4])]:
            pt = rng.choice(pool)
            if pt == chess.PAWN and chess.square_rank(s) in (0, 7):
                pt = chess.KNIGHT
            b.set_piece_at(s, chess.Piece(pt, rng.choice([chess.WHITE, chess.BLACK])))
        b.turn = rng.choice([chess.WHITE, chess.BLACK])
        if b.is_valid() and not b.is_game_over():
            return b.fen()


fens = [pos() for _ in range(n)]
cmds = [f"setoption name EvalFile value {os.environ.get('ONECHESS', '.')}/reseau/onechess.nnue",
        f"setoption name SyzygyPath value {TB}"]
for f in fens:
    cmds += ["ucinewgame", f"position fen {f}", "go depth 14", "isready"]
cmds.append("quit")
out = subprocess.run([engine], input="\n".join(cmds) + "\n", capture_output=True, text=True).stdout
bad_lines = 0
tot_lines = 0
bad_pos = 0
cur_bad = False
for line in out.splitlines():
    if line.startswith("info depth"):
        t = line.split()
        if "cp" in t:
            v = abs(int(t[t.index("cp") + 1]))
            tot_lines += 1
            if 13000 <= v < 19754 or v > 20000:
                bad_lines += 1
                cur_bad = True
    elif line.startswith("bestmove"):
        bad_pos += cur_bad
        cur_bad = False
print(f"{engine}: {bad_pos}/{n} positions avec au moins un score aberrant, "
      f"{bad_lines}/{tot_lines} lignes info")
