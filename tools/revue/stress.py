#!/usr/bin/env python3
"""Stress de la version de debogage : positions aleatoires (dont finales a peu
de pieces pour les tables), recherches courtes, plusieurs threads. Toute
assertion ou plantage arrete le moteur : on le detecte au code de sortie."""
import os
import random
import subprocess
import sys

import chess

engine = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 2
tb = sys.argv[4] if len(sys.argv) > 4 else ""
rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 3)


def rand_pos():
    b = chess.Board()
    target = rng.choice([32, 20, 12, 8, 6, 5, 4])
    for _ in range(400):
        if len(b.piece_map()) <= target or b.is_game_over():
            break
        ms = list(b.legal_moves)
        caps = [m for m in ms if b.is_capture(m)]
        b.push(rng.choice(caps) if caps and rng.random() < 0.6 else rng.choice(ms))
    return None if b.is_game_over() else b.fen()


fens = []
while len(fens) < n:
    f = rand_pos()
    if f:
        fens.append(f)

cmds = [f"setoption name EvalFile value {os.environ.get('ONECHESS', '.')}/reseau/onechess.nnue",
        f"setoption name Threads value {threads}"]
if tb:
    cmds.append(f"setoption name SyzygyPath value {tb}")
for f in fens:
    cmds += [f"position fen {f}", f"go nodes {rng.choice([2000, 20000, 60000])}"]
    cmds.append("isready")
cmds.append("quit")
p = subprocess.run([engine], input="\n".join(cmds) + "\n", capture_output=True, text=True, timeout=3600)
best = sum(1 for l in p.stdout.splitlines() if l.startswith("bestmove"))
print(f"code {p.returncode}  bestmove {best}/{n}")
if p.returncode != 0 or best != n:
    print(p.stderr[-2000:])
    print("\n".join(p.stdout.splitlines()[-5:]))
