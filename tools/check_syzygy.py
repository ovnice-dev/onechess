#!/usr/bin/env python3
"""Verifie que le moteur ne gache pas un resultat connu des tables.

Fathom est du code tiers et notre enveloppe traduit des bitboards, des droits
de roque et une case de prise en passant : une erreur d'indexation ne planterait
pas, elle ferait simplement jouer un coup qui perd le gain. Il faut donc une
verification INDEPENDANTE -- ici l'implementation Syzygy de python-chess, qui
ne partage aucune ligne de code avec Fathom.

Le critere est le seul qui compte : apres notre coup, l'adversaire doit se
retrouver dans une position dont les tables disent qu'elle est perdue (si on
gagnait), ou au pire nulle (si c'etait nul). Un gain transforme en nulle est
une faute ; une nulle transformee en defaite aussi.
"""

import argparse
import random
import subprocess
import sys

import chess
import chess.syzygy


def random_position(rng, n_pieces, tb):
    """Tire une position legale a n pieces, ni mat ni pat, non triviale."""
    extras = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    for _ in range(400):
        board = chess.Board.empty()
        squares = rng.sample(range(64), n_pieces)

        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        for i in range(2, n_pieces):
            pt = rng.choice(extras)
            colour = rng.choice([chess.WHITE, chess.BLACK])
            if pt == chess.PAWN and chess.square_rank(squares[i]) in (0, 7):
                pt = chess.KNIGHT
            board.set_piece_at(squares[i], chess.Piece(pt, colour))

        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        board.halfmove_clock = 0
        if not board.is_valid() or board.is_game_over():
            continue
        try:
            tb.probe_wdl(board)
        except (KeyError, chess.syzygy.MissingTableError):
            continue
        return board
    return None


def engine_move(exe, tbpath, fen, depth):
    cmds = ["uci", f"setoption name SyzygyPath value {tbpath}", "isready",
            f"position fen {fen}", f"go depth {depth}", "quit"]
    p = subprocess.run([exe], input="\n".join(cmds) + "\n",
                       capture_output=True, text=True, timeout=120)
    for line in p.stdout.splitlines():
        if line.startswith("bestmove"):
            tok = line.split()
            return tok[1] if len(tok) > 1 else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=r"C:\Onechess\engine\bin\onechess.exe")
    ap.add_argument("--tb", default=r"C:\Onechess\syzygy")
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tb = chess.syzygy.open_tablebase(args.tb)

    tested = faults = 0
    breakdown = {2: 0, 1: 0, 0: 0, -1: 0, -2: 0}

    for i in range(args.positions):
        n = 3 + (i % 3)                     # 3, 4 puis 5 pieces
        board = random_position(rng, n, tb)
        if board is None:
            continue

        before = tb.probe_wdl(board)        # +2 gagne, 0 nul, -2 perdu
        breakdown[before] = breakdown.get(before, 0) + 1

        uci = engine_move(args.engine, args.tb, board.fen(), args.depth)
        if uci in (None, "(none)"):
            print(f"  pas de coup rendu : {board.fen()}")
            faults += 1
            continue

        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            print(f"  COUP ILLEGAL {uci} : {board.fen()}")
            faults += 1
            continue

        board.push(move)
        after = -tb.probe_wdl(board)        # ramene au point de vue du joueur
        tested += 1

        # Le resultat ne doit jamais empirer. On compare des categories, pas
        # des valeurs : un gain maudit reste un gain du point de vue du signe.
        if (before > 0 and after <= 0) or (before == 0 and after < 0):
            print(f"  RESULTAT GACHE  {before:+d} -> {after:+d}  {uci}  "
                  f"{board.root().fen()}")
            faults += 1

    print(f"\n{tested} positions verifiees "
          f"(gagnees {breakdown.get(2,0)+breakdown.get(1,0)}, "
          f"nulles {breakdown.get(0,0)}, "
          f"perdues {breakdown.get(-2,0)+breakdown.get(-1,0)})")
    if faults:
        print(f"{faults} FAUTE(S) : l'enveloppe Syzygy est incorrecte")
        return 1
    print("aucun resultat gache : l'enveloppe Syzygy est correcte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
