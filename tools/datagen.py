#!/usr/bin/env python3
"""Generation de donnees d'entrainement NNUE.

Stockfish joue contre lui-meme a nombre de noeuds fixe, en partant d'ouvertures
tirees au hasard. Chaque position visitee est enregistree avec le score que la
recherche vient de calculer, puis, en fin de partie, avec le resultat.

Une seule recherche sert donc a deux choses : choisir le coup et etiqueter la
position. C'est ce qui rend l'operation abordable sur un processeur de bureau.

Le dialogue UCI passe par des tuyaux bruts et non par la couche moteur de
python-chess : celle-ci ajoute environ 2,6 ms par appel, soit les deux tiers du
temps quand la recherche elle-meme ne dure que 1,3 ms.

Format d'enregistrement, 32 octets, petit-boutiste :

    uint64 occ        cases occupees
    uint8  pcs[16]    4 bits par piece, dans l'ordre croissant des cases
    int16  score      centipions du point de vue des BLANCS
    uint8  stm        0 = trait aux blancs
    uint8  result     0 = noirs gagnent, 1 = nulle, 2 = blancs gagnent
    uint8  rule50
    uint8  pad[3]
"""

import argparse
import os
import random
import struct
import subprocess
import sys
import time
from multiprocessing import Process, Queue

import chess

# Le repertoire du script n'est pas toujours dans sys.path quand
# multiprocessing relance l'interpreteur par spawn.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import winjob

RECORD_SIZE = 32
_PACK = struct.Struct("<Q16shBBB3x")

# code piece = (type - 1) + 6 * couleur, soit 0..11 pour PNBRQK puis pnbrqk
_PIECE_CODE = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}


def pack_position(board, score_white, stm, result, rule50):
    occ = 0
    nibbles = []
    for sq in range(64):
        pc = board.piece_at(sq)
        if pc is not None:
            occ |= 1 << sq
            nibbles.append(_PIECE_CODE[(pc.piece_type, pc.color)])

    pcs = bytearray(16)
    for i, code in enumerate(nibbles):
        if i & 1:
            pcs[i >> 1] |= code << 4
        else:
            pcs[i >> 1] = code

    return _PACK.pack(occ, bytes(pcs), score_white, stm, result, min(rule50, 255))


# --------------------------------------------------------------------------
# Pilote UCI minimal
# --------------------------------------------------------------------------

class RawEngine:
    """Dialogue UCI direct, sans couche intermediaire."""

    def __init__(self, path, threads=1, hash_mb=16):
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, creationflags=flags)

        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        self.isready()

    def _send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def _wait_for(self, token):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("moteur termine prematurement")
            if line.startswith(token):
                return line

    def isready(self):
        self._send("isready")
        self._wait_for("readyok")

    def new_game(self):
        self._send("ucinewgame")
        self.isready()

    def search(self, fen, moves, nodes):
        """Rend (coup uci, score en cp vu du trait, mat ou non)."""
        pos = f"position fen {fen}"
        if moves:
            pos += " moves " + " ".join(moves)
        self._send(pos)
        self._send(f"go nodes {nodes}")

        score, is_mate = None, False
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("moteur termine prematurement")

            if line.startswith("bestmove"):
                return line.split()[1], score, is_mate

            if line.startswith("info") and " score " in line:
                t = line.split()
                i = t.index("score")
                if t[i + 1] == "cp":
                    score, is_mate = int(t[i + 2]), False
                elif t[i + 1] == "mate":
                    m = int(t[i + 2])
                    score, is_mate = (30000 - abs(m)) * (1 if m > 0 else -1), True

    def quit(self):
        try:
            self._send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


# --------------------------------------------------------------------------
# Un travailleur : des parties, des positions
# --------------------------------------------------------------------------

def worker(wid, args, openings, out_path, queue):
    rng = random.Random(args.seed + wid * 7919)
    eng = RawEngine(args.engine, threads=1, hash_mb=args.hash)

    written  = 0
    rejected = 0        # departs ecartes par le filtre d'equilibre
    buf = bytearray()

    with open(out_path, "wb") as f:
        while written < args.positions_per_worker:
            # Arret propre demande (limite de 12 h sur Kaggle) : on finit
            # entre deux parties, rien n'est perdu.
            if args.stop_file and os.path.exists(args.stop_file):
                break
            start_fen = rng.choice(openings)
            board = chess.Board(start_fen)

            # Coups aleatoires en plus du livre. C'est ce qui fait la diversite
            # du lot, et il en faut beaucoup : le livre ne contient que ~1400
            # positions, alors qu'un lot de 100 M demande ~3,3 M de parties.
            # Sans ces coups, chaque ouverture serait rejouee 2400 fois et le
            # reseau apprendrait une poignee de structures par coeur.
            for _ in range(rng.randint(args.random_plies // 2, args.random_plies)):
                legal = list(board.legal_moves)
                if not legal:
                    break
                board.push(rng.choice(legal))

            if board.is_game_over():
                continue

            eng.new_game()
            root_fen = board.fen()

            # Filtre d'equilibre. Des coups tires au hasard produisent parfois
            # une position deja perdue -- une dame en prise au 6e coup. La
            # partie qui suit n'apprend rien : elle est decidee d'avance, et
            # ses positions portent un score que le materiel explique a lui
            # seul. Une recherche courte suffit a les ecarter.
            if args.balance_cp:
                _, s0, m0 = eng.search(root_fen, [], args.balance_nodes)
                if s0 is None or m0 or abs(s0) > args.balance_cp:
                    rejected += 1
                    continue
            moves = []
            samples = []           # (bytes-sans-resultat, index du resultat)
            adj_run, adj_side = 0, None
            result = None

            for _ply in range(args.max_plies):
                if board.is_game_over(claim_draw=True):
                    r = board.result(claim_draw=True)
                    result = 2 if r == "1-0" else 0 if r == "0-1" else 1
                    break

                uci, score, is_mate = eng.search(root_fen, moves, args.nodes)
                if uci == "(none)" or score is None:
                    result = 1
                    break

                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    result = 1
                    break

                score_white = score if board.turn == chess.WHITE else -score

                # Filtres classiques : une position en echec ou dont le
                # meilleur coup est une prise n'a pas d'evaluation statique
                # exploitable -- le reseau apprendrait du bruit tactique.
                keep = (not board.is_check()
                        and not board.is_capture(move)
                        and move.promotion is None
                        and not is_mate
                        and abs(score_white) <= args.max_score)

                if keep:
                    samples.append(pack_position(
                        board, max(-10000, min(10000, score_white)),
                        0 if board.turn == chess.WHITE else 1,
                        1,                       # resultat provisoire
                        board.halfmove_clock))

                # Adjudication : inutile de jouer une finale deja gagnee
                if abs(score_white) >= args.adj_score:
                    side = 2 if score_white > 0 else 0
                    adj_run = adj_run + 1 if side == adj_side else 1
                    adj_side = side
                    if adj_run >= args.adj_plies:
                        result = adj_side
                        break
                else:
                    adj_run, adj_side = 0, None

                board.push(move)
                moves.append(uci)
            else:
                result = 1

            if result is None:
                result = 1

            # Le resultat est le 27e octet de chaque enregistrement
            for rec in samples:
                buf += rec[:27] + bytes([result]) + rec[28:]

            written += len(samples)

            # Tampon volontairement petit. A 1 Mo -- 32 768 positions par
            # travailleur -- une campagne interrompue au bout de 23 minutes
            # n'avait RIEN ecrit sur disque : tout etait encore en memoire, et
            # tout a ete perdu. 64 Ko bornent la perte a 2048 positions par
            # travailleur, et rendent la progression visible des la premiere
            # minute. Le cout en ecritures est negligeable devant le temps de
            # recherche.
            if len(buf) >= 1 << 16:
                f.write(buf)
                f.flush()
                buf.clear()
                queue.put(("progress", wid, written))

        if buf:
            f.write(buf)

    eng.quit()
    queue.put(("done", wid, written, rejected))


def load_openings(path):
    fens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                fens.append(line)
    if not fens:
        raise SystemExit(f"aucune ouverture dans {path}")
    return fens


def main():
    winjob.install()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default=os.path.expandvars(r"%APPDATA%")
                                        + r""
                                        r"\org.encroissant.app\engines\stockfish"
                                        r"\stockfish-windows-x86-64-avx2.exe")
    ap.add_argument("--openings", default=r"C:\Onechess\data\openings.epd")
    ap.add_argument("--out", default=r"C:\Onechess\data\nnue")
    ap.add_argument("--positions", type=int, default=20_000_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--random-plies", type=int, default=10,
                    help="coups aleatoires apres le livre ; tires entre la "
                         "moitie de ce nombre et ce nombre")
    ap.add_argument("--balance-cp", type=int, default=400,
                    help="ecarte un depart deja desequilibre ; 0 = desactive")
    ap.add_argument("--balance-nodes", type=int, default=3000)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--max-score", type=int, default=3000)
    ap.add_argument("--adj-score", type=int, default=2500)
    ap.add_argument("--adj-plies", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stop-file", default="",
                    help="fichier dont l'apparition arrete proprement la generation")
    args = ap.parse_args()

    args.positions_per_worker = args.positions // args.workers
    os.makedirs(args.out, exist_ok=True)
    openings = load_openings(args.openings)

    print(f"{args.workers} travailleurs   {args.nodes} noeuds/coup   "
          f"cible {args.positions:,} positions   sortie {args.out}")

    q = Queue()
    procs = []
    for w in range(args.workers):
        # Nom horodate : une relance dans le meme dossier AJOUTE au lot au lieu
        # d'ecraser shard00.bin & co. C'est exactement le defaut qui a detruit
        # 2 769 764 positions d'etiquetage le 05-09 ; label.py en avait ete
        # corrige, datagen.py pas encore.
        out_path = os.path.join(args.out, f"auto_{time.strftime('%m%d_%H%M%S')}_{w:02d}.bin")
        p = Process(target=worker, args=(w, args, openings, out_path, q))
        p.start()
        procs.append(p)

    counts  = [0] * args.workers
    rejects = [0] * args.workers
    done = 0
    t0 = time.time()
    last = t0

    while done < args.workers:
        msg = q.get()
        kind, wid, n = msg[0], msg[1], msg[2]
        counts[wid] = n
        if kind == "done":
            done += 1
            rejects[wid] = msg[3]
        now = time.time()
        if now - last > 10 or kind == "done":
            last = now
            total = sum(counts)
            rate = total / max(1e-9, now - t0)
            eta = (args.positions - total) / rate / 3600 if rate > 0 else 0
            print(f"  {total:>12,} positions   {rate:7.0f}/s   reste {eta:5.2f} h",
                  flush=True)

    for p in procs:
        p.join()

    total = sum(counts)
    dt = time.time() - t0
    print(f"\ntermine : {total:,} positions en {dt/3600:.2f} h "
          f"({total/dt:.0f}/s, {total*RECORD_SIZE/1e9:.2f} Go)")
    if sum(rejects):
        print(f"filtre d'equilibre : {sum(rejects):,} departs ecartes")


if __name__ == "__main__":
    main()
