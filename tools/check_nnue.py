#!/usr/bin/env python3
"""Verifie que l'inference C++ reproduit exactement le reseau entraine.

Rejoue en numpy le calcul entier tel que le fait le moteur -- meme
quantification, memes bornes, meme indexation avec buckets et miroir -- puis
compare a ce que le moteur rend sur les memes positions.

C'est la seule facon d'attraper une erreur d'indexation ou de perspective :
un reseau mal branche ne plante jamais, il joue simplement moins bien, et il
faudrait des heures de match pour s'en apercevoir.
"""

import argparse
import struct
import subprocess
import sys

import chess
import numpy as np

BASE_INPUTS = 768
PAIR_INPUTS = 96 * 95 // 2   # paires de pions (PP_3Wide)
PAIRS_FLAG  = 256
QA, QB, SCALE = 255, 64, 400
MAGIC3 = 0x334E4E4F     # sans champ activation : CReLU
MAGIC4 = 0x344E4E4F     # avec champ activation


def tdiv(a, b):
    """Division entiere tronquee VERS ZERO, comme en C.

    Le // de Python arrondit vers le bas : sur un score negatif il
    donnerait un centipion d'ecart avec le moteur, et le test crierait
    a l'erreur d'indexation pour une simple convention d'arrondi.
    """
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 10",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "8/8/8/8/8/6k1/6p1/6K1 w - - 0 1",
    "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
    "5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1",
    "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
    "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1",
    "1r3k2/4q3/2Pp3b/3Bp3/2Q2p2/1p1P2P1/1P2KP2/3N4 w - - 0 1",
    # Rois sur les deux moities, pour exercer le miroir et les buckets
    "r4rk1/pppq1ppp/2n1bn2/3p4/3P4/2N1BN2/PPPQ1PPP/R4RK1 w - - 0 1",
    "2kr3r/pppq1ppp/2n1bn2/3p4/3P4/2N1BN2/PPPQ1PPP/2KR3R b - - 0 1",
    "8/8/4k3/8/8/3K4/8/8 w - - 0 1",
    "8/8/8/8/8/8/6k1/6K1 b - - 0 1",
]


class Net:
    def __init__(self, path):
        with open(path, "rb") as f:
            magic, hl, nb, mirror, ob_n = struct.unpack("<IIIII", f.read(20))
            if magic not in (MAGIC3, MAGIC4):
                raise SystemExit("fichier reseau invalide (magic)")
            self.act = struct.unpack("<I", f.read(4))[0] if magic == MAGIC4 else 0
            self.pairs = bool(self.act & PAIRS_FLAG)
            self.act &= ~PAIRS_FLAG
            self.hl, self.nb, self.mirror, self.ob_n = hl, nb, mirror == 1, ob_n
            self.material = mirror == 2   # experts choisis par le materiel
            self.layout = np.frombuffer(f.read(64), dtype=np.uint8).astype(int)
            nfeat = nb * BASE_INPUTS
            self.fw = np.frombuffer(f.read(nfeat * hl * 2),
                                    dtype="<i2").reshape(nfeat, hl).astype(np.int32)
            self.fb = np.frombuffer(f.read(hl * 2), dtype="<i2").astype(np.int32)
            self.pw = (np.frombuffer(f.read(PAIR_INPUTS * hl * 2), dtype="<i2")
                       .reshape(PAIR_INPUTS, hl).astype(np.int32) if self.pairs else None)
            self.ow = np.frombuffer(f.read(ob_n * 2 * hl * 2),
                                    dtype="<i2").reshape(ob_n, 2 * hl).astype(np.int32)
            self.ob = np.frombuffer(f.read(ob_n * 4), dtype="<i4").astype(np.int64)

    def persp_info(self, persp, ksq):
        rel = ksq if persp == chess.WHITE else ksq ^ 56
        flip = 7 if (self.mirror and (rel & 7) >= 4) else 0
        rel ^= flip
        return self.layout[rel], flip

    def features(self, board, persp):
        ksq = board.king(persp)
        bucket, flip = self.persp_info(persp, ksq)
        if self.material:
            bucket, flip = min(max((len(board.piece_map()) - 2) // 8, 0), self.nb - 1), 0
        out = []
        for sq, pc in board.piece_map().items():
            colour = 0 if pc.color == chess.WHITE else 1
            p = 0 if persp == chess.WHITE else 1
            pidx = (1 if colour != p else 0) * 6 + (pc.piece_type - 1)
            rel = (sq if persp == chess.WHITE else sq ^ 56) ^ flip
            out.append(bucket * BASE_INPUTS + pidx * 64 + rel)
        return out

    def pair_features(self, board, persp):
        """Paires de pions (meme colonne ou colonnes voisines), cote Python."""
        pawns = [(sq, pc.color) for sq, pc in board.piece_map().items()
                 if pc.piece_type == chess.PAWN]
        out = []
        for i in range(len(pawns)):
            for j in range(i + 1, len(pawns)):
                (a, ca), (b, cb) = pawns[i], pawns[j]
                if abs((a & 7) - (b & 7)) > 1:
                    continue
                ids = []
                for sq, c in ((a, ca), (b, cb)):
                    rel = sq if persp == chess.WHITE else sq ^ 56
                    ids.append(48 * (c != persp) + rel - 8)
                hi, lo = max(ids), min(ids)
                out.append(hi * (hi - 1) // 2 + lo)
        return out

    def evaluate(self, fen):
        board = chess.Board(fen)
        acc = {}
        for persp in (chess.WHITE, chess.BLACK):
            a = self.fb.copy()
            for f in self.features(board, persp):
                a = a + self.fw[f]
            if self.pairs:
                for f in self.pair_features(board, persp):
                    a = a + self.pw[f]
            # Le moteur garde l'accumulateur en int16 : on reproduit la troncature
            acc[persp] = a.astype(np.int16).astype(np.int32)

        stm = board.turn
        us   = np.clip(acc[stm], 0, QA).astype(np.int64)
        them = np.clip(acc[not stm], 0, QA).astype(np.int64)
        if self.act == 1:                      # SCReLU
            us, them = us * us, them * them

        # Meme formule de bucket qu'en C++ et dans l'entraineur
        b = (len(board.piece_map()) - 2) // (32 // self.ob_n)
        w = self.ow[b]

        total = int((us * w[:self.hl]).sum() + (them * w[self.hl:]).sum())
        if self.act == 1:
            total = tdiv(total, QA)            # sortie en QA^2, biais en QA
        cp = tdiv((total + int(self.ob[b])) * SCALE, QA * QB)
        return max(-12000, min(12000, cp))


def eval_engine(engine, netpath, fens):
    cmds = [f"setoption name EvalFile value {netpath}", "isready"]
    for fen in fens:
        cmds += [f"position fen {fen}", "eval"]
    cmds.append("quit")

    p = subprocess.run([engine], input="\n".join(cmds) + "\n",
                       capture_output=True, text=True, timeout=180)
    vals = [int(l.split(":")[1].strip().split()[0])
            for l in p.stdout.splitlines() if l.startswith("Evaluation statique")]
    return vals, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=r"C:\Onechess\engine\bin\onechess.nnue")
    ap.add_argument("--engine", default=r"C:\Onechess\engine\bin\onechess.exe")
    args = ap.parse_args()

    net = Net(args.net)
    print(f"reseau : {net.nb} x 768 x {net.hl} x2 -> {net.ob_n}   "
          f"miroir={int(net.mirror)}   "
          f"{'screlu' if net.act else 'crelu'}")

    ref = [net.evaluate(f) for f in FENS]
    got, raw = eval_engine(args.engine, args.net, FENS)

    if len(got) != len(FENS):
        print("le moteur n'a pas rendu le bon nombre d'evaluations :")
        print(raw[:2000])
        return 1

    bad = 0
    for fen, r, g in zip(FENS, ref, got):
        ok = (r == g)
        bad += not ok
        print(f"  {'OK  ' if ok else 'ECART'}  numpy {r:6d}   moteur {g:6d}   {fen[:44]}")

    print()
    if bad:
        print(f"{bad}/{len(FENS)} positions divergent : l'inference C++ ne reproduit "
              f"pas le reseau entraine")
        return 1
    print(f"{len(FENS)}/{len(FENS)} positions identiques : l'inference C++ est exacte")

    # Le reseau rend toujours le score DU CAMP AU TRAIT. Une position et son
    # miroir couleur sont la meme position pour celui qui joue : les deux
    # doivent donner la MEME valeur. Un ecart signalerait une perspective ou
    # un miroir de case errone.
    print("\nSymetrie (position retournee = meme valeur) :")
    worst = 0
    for fen in FENS[:8]:
        b = chess.Board(fen)
        a, c = net.evaluate(b.fen()), net.evaluate(b.mirror().fen())
        worst = max(worst, abs(a - c))
        print(f"  {a:6d}  /  {c:6d}   ecart {a - c:+4d}")
    print(f"ecart maximal {worst} cp "
          f"({'symetrie exacte' if worst <= 1 else 'ATTENTION : asymetrie'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
