#!/usr/bin/env python3
"""Compare des reseaux sur UN MEME jeu de validation.

L'erreur affichee par train.py n'est comparable qu'entre reseaux entraines sur
la meme distribution : chacun est valide sur SA propre part de validation, et
deux distributions n'ont pas la meme difficulte intrinseque. Le 09-09, lire ces
chiffres tels quels aurait fait conclure a l'envers sur les donnees Lichess
(96,4 cp contre 63,7 : le premier chiffre mesurait une distribution plus dure,
pas un reseau plus mauvais).

Ce script tire N positions d'un fichier, les fait evaluer statiquement par
chaque reseau via la commande `eval` du moteur, et compare au score stocke.

LE FICHIER DOIT ETRE TENU A L'ECART DE TOUS LES RESEAUX COMPARES. Une position
vue a l'entrainement flatte le reseau qui l'a vue. La comparaison du 10-09 en
a souffert : le fichier retenu figurait dans l'entrainement du reseau installe
et du melange, pas dans celui des autres. Les conclusions tirees deux a deux
tenaient (chaque paire comparee etait dans la meme situation), mais le plus
sur est un fichier reserve des le depart et jamais inclus dans aucun
entrainement -- typiquement un fichier d'autojeu mis de cote.

Exemple :
    python tools/val_commun.py --data data/nnue_auto_val/shard08.bin \\
        --nets engine/bin/onechess.nnue engine/bin/G_auto.nnue
"""

import argparse
import os
import struct
import subprocess
import sys

import chess
import numpy as np

_P = struct.Struct("<Q16shBBB3x")
CODE = "PNBRQKpnbrqk"


def decode(rec):
    """Enregistrement 32 octets -> (position, score vu du TRAIT).

    Le score est stocke du point de vue des blancs ; `eval` rend celui du camp
    au trait. On ramene donc la cible au trait, sans quoi une position sur deux
    aurait sa cible inversee.
    """
    occ, pcs, sc, stm, _res, _r50 = _P.unpack(rec)
    b = chess.Board.empty()
    n = 0
    for sq in range(64):
        if occ >> sq & 1:
            code = pcs[n >> 1] & 0x0F if not (n & 1) else pcs[n >> 1] >> 4
            b.set_piece_at(sq, chess.Piece.from_symbol(CODE[code]))
            n += 1
    b.turn = chess.WHITE if stm == 0 else chess.BLACK
    return b, (sc if stm == 0 else -sc)


def evaluate(engine, net, fens):
    cmds = [f"setoption name EvalFile value {net}", "isready"]
    for f in fens:
        cmds += [f"position fen {f}", "eval"]
    cmds.append("quit")
    out = subprocess.run([engine], input="\n".join(cmds) + "\n",
                         capture_output=True, text=True, timeout=900).stdout
    return [int(line.split(":")[1].strip().split()[0]) for line in out.splitlines()
            if line.startswith("Evaluation statique")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="fichier .bin tenu a l'ecart de TOUS les reseaux compares")
    ap.add_argument("--nets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", default=r"C:\Onechess\engine\bin\onechess.exe")
    args = ap.parse_args()

    raw = open(args.data, "rb").read()
    total = len(raw) // 32
    if total == 0:
        raise SystemExit(f"{args.data} est vide")

    rng = np.random.default_rng(args.seed)
    fens, tgt = [], []
    for i in rng.choice(total, min(total, args.n * 2), replace=False):
        b, t = decode(raw[int(i) * 32:(int(i) + 1) * 32])
        if b.is_valid() and not b.is_game_over():
            fens.append(b.fen())
            tgt.append(t)
        if len(fens) >= args.n:
            break
    tgt = np.array(tgt, dtype=float)

    print(f"{len(fens)} positions tirees de {os.path.basename(args.data)} "
          f"({total:,} au total), cible vue du trait\n")
    print(f"  {'reseau':28s} {'erreur':>10s} {'mediane':>10s} {'correlation':>12s}")
    for net in args.nets:
        p = np.array(evaluate(args.engine, net, fens), dtype=float)
        if len(p) != len(fens):
            print(f"  {os.path.basename(net):28s} ECHEC : {len(p)} evaluations "
                  f"pour {len(fens)} positions")
            continue
        e = np.abs(p - tgt)
        print(f"  {os.path.basename(net):28s} {e.mean():7.1f} cp {np.median(e):7.1f} cp "
              f"{np.corrcoef(p, tgt)[0, 1]:+12.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
