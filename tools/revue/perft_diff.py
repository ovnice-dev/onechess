#!/usr/bin/env python3
"""Perft differentiel : moteur contre python-chess, sur des positions tirees
au hasard (echecs classiques et Chess960)."""
import random
import subprocess
import sys

import chess

ENGINE = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rng = random.Random(7)


def perft(b, d):
    if d == 1:
        return b.legal_moves.count()
    n = 0
    for m in b.legal_moves:
        b.push(m)
        n += perft(b, d - 1)
        b.pop()
    return n


def random_position(c960):
    if c960:
        b = chess.Board.from_chess960_pos(rng.randrange(960))
    else:
        b = chess.Board()
    plies = rng.randrange(0, 80)
    for _ in range(plies):
        moves = list(b.legal_moves)
        if not moves:
            break
        # favoriser roques, prises, promotions, poussees doubles
        spec = [m for m in moves if b.is_castling(m) or b.is_capture(m) or m.promotion
                or (b.piece_type_at(m.from_square) == chess.PAWN and abs(m.to_square - m.from_square) == 16)]
        m = rng.choice(spec) if spec and rng.random() < 0.5 else rng.choice(moves)
        b.push(m)
    return b


cases = []
while len(cases) < N:
    c960 = len(cases) % 2 == 1
    b = random_position(c960)
    if b.is_game_over():
        continue
    fen = b.fen(shredder=c960) if c960 else b.fen()
    cases.append((c960, fen, perft(b, DEPTH)))

for c960 in (False, True):
    sel = [c for c in cases if c[0] == c960]
    cmds = [f"setoption name UCI_Chess960 value {'true' if c960 else 'false'}"]
    for _, fen, _ in sel:
        cmds += [f"position fen {fen}", f"perft {DEPTH}"]
    cmds.append("quit")
    out = subprocess.run([ENGINE], input="\n".join(cmds) + "\n", capture_output=True, text=True).stdout
    got = [int(l.split()[1]) for l in out.splitlines() if l.startswith("Noeuds:")]
    bad = 0
    for (_, fen, ref), g in zip(sel, got):
        if ref != g:
            bad += 1
            print(f"ECART {'960' if c960 else 'std'} {fen}: python-chess {ref}, moteur {g}")
    print(f"{'Chess960' if c960 else 'classique'} : {len(got)}/{len(sel)} positions, {bad} ecarts")
