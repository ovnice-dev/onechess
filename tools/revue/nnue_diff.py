#!/usr/bin/env python3
"""Compare l'evaluation du moteur a la reference numpy de tools/check_nnue.py
sur beaucoup de positions (aleatoires + celles de check_nnue)."""
import os
import random
import subprocess
import sys

import chess

sys.path.insert(0, os.path.join(os.environ.get("ONECHESS", "."), "tools"))
import check_nnue as cn  # noqa: E402

engine, net_path = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
rng = random.Random(11)

fens = list(cn.FENS)
while len(fens) < n:
    b = chess.Board()
    for _ in range(rng.randrange(0, 140)):
        ms = list(b.legal_moves)
        if not ms:
            break
        caps = [m for m in ms if b.is_capture(m)]
        b.push(rng.choice(caps) if caps and rng.random() < 0.4 else rng.choice(ms))
    if not b.is_game_over():
        fens.append(b.fen())

net = cn.Net(net_path)
ref = [net.evaluate(f) for f in fens]
got, raw = cn.eval_engine(engine, net_path, fens)
if len(got) != len(fens):
    print("sortie moteur incomplete :", raw[:500])
    sys.exit(2)
bad = [(f, r, g) for f, r, g in zip(fens, ref, got) if r != g]
print(f"{net_path}: {len(fens) - len(bad)}/{len(fens)} identiques")
for f, r, g in bad[:8]:
    print(f"   ECART numpy {r} moteur {g}  {f}")
sys.exit(1 if bad else 0)
