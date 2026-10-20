#!/usr/bin/env python3
"""Etiquette des positions avec Stockfish.

Prend un CSV "fen,result" (positions de parties humaines) et produit le format
binaire 32 octets attendu par l'entrainement NNUE, en remplacant l'etiquette
faible qu'est le resultat d'une partie humaine par le score d'une vraie
recherche.

Le resultat de la partie est conserve a cote du score : l'entrainement melange
les deux, parce que le score dit ou en est la position et le resultat dit ce
qu'elle devient.

Filtres appliques, dans l'ordre du moins cher au plus cher :
  - numero de coup minimal, pour jeter la partie d'ouverture qui se repete
    d'une partie a l'autre et n'apprend rien au reseau ;
  - position en echec, car le moteur n'evalue jamais statiquement une telle
    position : la recherche y genere des parades ;
  - score trop grand, ou le reseau n'a plus rien a apprendre.
"""

import argparse
import datetime
import os
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

_PIECE_CODE = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}

# Table de conversion des caracteres FEN vers le code piece, pour empaqueter
# sans construire d'objet Board.
_FEN_CODE = {c: i for i, c in enumerate("PNBRQKpnbrqk")}


def pack_from_fen(fen_board, stm, score_white, result, rule50):
    """Empaquette directement depuis la partie plateau du FEN."""
    occ = 0
    codes = []
    sq = 56                       # le FEN commence par la 8e rangee
    for ch in fen_board:
        if ch == "/":
            sq -= 16
        elif ch.isdigit():
            sq += int(ch)
        else:
            occ |= 1 << sq
            codes.append((sq, _FEN_CODE[ch]))
            sq += 1

    codes.sort()                  # ordre croissant des cases, comme occ
    pcs = bytearray(16)
    for i, (_, code) in enumerate(codes):
        if i & 1:
            pcs[i >> 1] |= code << 4
        else:
            pcs[i >> 1] = code

    return _PACK.pack(occ, bytes(pcs), score_white, stm, result, min(rule50, 255))


class RawEngine:
    def __init__(self, path, threads=1, hash_mb=16):
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, creationflags=flags)
        self._send("uci")
        self._wait("uciok")
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("isready")
        self._wait("readyok")

    def _send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def _wait(self, token):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("moteur termine")
            if line.startswith(token):
                return line

    def score(self, fen, nodes):
        """Rend (score en cp vu du trait, meilleur coup) ; score None si mat."""
        self._send(f"position fen {fen}")
        self._send(f"go nodes {nodes}")
        cp = None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("moteur termine")
            if line.startswith("bestmove"):
                parts = line.split()
                return cp, (parts[1] if len(parts) > 1 else None)
            if line.startswith("info") and " score cp " in line:
                t = line.split()
                cp = int(t[t.index("score") + 2])
            elif line.startswith("info") and " score mate " in line:
                cp = None

    def quit(self):
        try:
            self._send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def worker(wid, args, in_q, out_path, stat_q):
    eng = RawEngine(args.engine, threads=1, hash_mb=args.hash)
    kept = seen = 0
    buf = bytearray()

    with open(out_path, "wb") as f:
        while True:
            batch = in_q.get()
            if batch is None:
                break

            for line in batch:
                seen += 1
                fen, _, res = line.rpartition(",")
                parts = fen.split(" ")
                if len(parts) < 6:
                    continue

                board_part, stm_c, _cast, _ep, r50, fullmove = parts[:6]

                if int(fullmove) < args.min_fullmove:
                    continue

                # En echec : le moteur n'evalue jamais statiquement une telle
                # position, la recherche y genere des parades.
                board = chess.Board(fen)
                if board.is_check():
                    continue

                cp, best = eng.score(fen, args.nodes)
                if cp is None or not best or best == "(none)":
                    continue

                # Position tactique : si le meilleur coup est une prise ou une
                # promotion, le score reflete un gain de materiel qu'aucune
                # evaluation STATIQUE ne peut representer. Mesure : le plancher
                # d'erreur passe de 63,5 a 50,6 cp en ne gardant que les
                # positions tranquilles, pour 18 % de positions en moins.
                if args.quiet_only:
                    try:
                        mv = chess.Move.from_uci(best)
                    except ValueError:
                        continue
                    if mv not in board.legal_moves:
                        continue
                    if board.is_capture(mv) or mv.promotion is not None:
                        continue

                score_white = cp if stm_c == "w" else -cp
                if abs(score_white) > args.max_score:
                    continue

                r = res.strip()
                result = 2 if r == "1.0" else 0 if r == "0.0" else 1

                buf += pack_from_fen(board_part, 0 if stm_c == "w" else 1,
                                     score_white, result, int(r50))
                kept += 1

            if len(buf) >= 1 << 20:
                f.write(buf)
                buf.clear()
                stat_q.put((wid, kept, seen))

        if buf:
            f.write(buf)

    eng.quit()
    stat_q.put((wid, kept, seen))
    stat_q.put(("done", wid, 0))


def main():
    winjob.install()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=r"E:\training_data.csv")
    ap.add_argument("--engine", default=os.path.expandvars(r"%APPDATA%")
                                        + r""
                                        r"\org.encroissant.app\engines\stockfish"
                                        r"\stockfish-windows-x86-64-avx2.exe")
    ap.add_argument("--out", default=r"C:\Onechess\data\nnue")
    ap.add_argument("--limit", type=int, default=0, help="lignes lues, 0 = tout")
    ap.add_argument("--skip", type=int, default=0,
                    help="lignes sautees au debut, pour reprendre sans doublon")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nodes", type=int, default=2000)
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--min-fullmove", type=int, default=8)
    ap.add_argument("--max-score", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--quiet-only", type=int, default=1,
                    help="ne garder que les positions dont le meilleur coup est tranquille")
    ap.add_argument("--stop-file",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "data", "STOP_LABEL"),
                    help="si ce fichier apparait : arret propre, reprise exacte avec --skip")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Les fichiers portent un marqueur de campagne. Sans cela une reprise dans
    # le meme repertoire ouvrirait les memes noms en "wb", ce qui TRONQUE :
    # c'est exactement comme cela que 2,77 M positions ont ete detruites.
    tag = datetime.datetime.now().strftime("%m%d_%H%M")
    existing = [f for f in os.listdir(args.out) if f.startswith(f"lab_{tag}_")]
    if existing:
        raise SystemExit(f"une campagne porte deja le marqueur {tag} dans {args.out}")

    deja = sum(os.path.getsize(os.path.join(args.out, f))
               for f in os.listdir(args.out) if f.endswith(".bin"))
    if deja:
        print(f"{deja // RECORD_SIZE:,} positions deja presentes, conservees "
              f"(nouvelle campagne : lab_{tag}_*.bin)")

    in_q = Queue(maxsize=args.workers * 8)
    stat_q = Queue()
    procs = []
    for w in range(args.workers):
        p = Process(target=worker,
                    args=(w, args, in_q,
                          os.path.join(args.out, f"lab_{tag}_{w:02d}.bin"), stat_q))
        p.start()
        procs.append(p)

    print(f"{args.workers} travailleurs   {args.nodes} noeuds   source {args.csv}", flush=True)

    kept = [0] * args.workers
    seen = [0] * args.workers
    t0 = time.time()
    last = t0
    done = 0

    def drain(block=False):
        nonlocal done, last
        while True:
            try:
                m = stat_q.get_nowait() if not block else stat_q.get()
                block = False
            except Exception:
                return
            if m[0] == "done":
                done += 1
            else:
                wid, k, s = m
                kept[wid], seen[wid] = k, s
            now = time.time()
            if now - last > 15:
                last = now
                tk, ts = sum(kept), sum(seen)
                dt = now - t0
                print(f"  {tk:>11,} gardees / {ts:>11,} lues   {tk/dt:6.0f}/s   "
                      f"{100*tk/max(1,ts):4.1f} % gardees", flush=True)

    fed = 0
    stopped = False
    with open(args.csv, encoding="utf-8", errors="replace") as f:
        f.readline()
        for _ in range(args.skip):
            if not f.readline():
                break
        batch = []
        for line in f:
            batch.append(line)
            fed += 1
            if len(batch) >= args.batch:
                in_q.put(batch)
                batch = []
                drain()
                # Arret propre : on cesse de distribuer. Tout ce qui est deja
                # distribue sera traite, donc les lignes faites sont exactement
                # les `fed` premieres : une reprise avec --skip ne cree aucun
                # doublon et ne perd rien. Tuer le processus, au contraire,
                # laisse un melange de lots faits et non faits, impossible a
                # reprendre.
                if args.stop_file and os.path.exists(args.stop_file):
                    stopped = True
                    break
            if args.limit and fed >= args.limit:
                break
        if batch:
            in_q.put(batch)

    for _ in procs:
        in_q.put(None)

    while done < args.workers:
        drain(block=True)

    for p in procs:
        p.join()

    tk, ts = sum(kept), sum(seen)
    dt = time.time() - t0
    print(f"\ntermine : {tk:,} positions gardees sur {ts:,} lues en {dt/60:.1f} min "
          f"({tk/dt:.0f}/s, {tk*RECORD_SIZE/1e9:.2f} Go)")

    if stopped:
        reprise = args.skip + fed
        with open(os.path.join(args.out, "REPRISE"), "w", encoding="utf-8") as fh:
            fh.write(f"{reprise}\n")
        print(f"ARRET PROPRE : {fed:,} lignes traitees ; reprendre avec --skip {reprise}",
              flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
