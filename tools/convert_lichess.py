#!/usr/bin/env python3
"""Convertit les evaluations Lichess vers notre format 32 octets.

Source : Lichess/chess-position-evaluations, 394 M de positions uniques deja
evaluees par Stockfish, profondeur mediane 25 -- bien plus profond que nos
1000 noeuds, et gratuit.

TROIS PIEGES, dont un qui empoisonnerait tout le lot en silence :

1. MULTIPV. La meme position apparait plusieurs fois, une ligne par coup
   candidat, avec le score de CE coup. Seule la premiere ligne d'une suite
   porte l'evaluation de la position ; les suivantes sont les scores de coups
   inferieurs. Les prendre toutes apprendrait au reseau que les mauvais coups
   valent leur propre score. Les lignes d'une meme position sont consecutives :
   un simple "garder la premiere de chaque suite" suffit, sans table de hachage
   ni memoire.

2. LE SCORE EST DU POINT DE VUE DES BLANCS. Notre format aussi -- verifie sur
   une position ou les noirs ont le trait : la meilleure ligne y porte le cp le
   plus BAS, donc le plus favorable aux noirs.

3. LE FEN EST TRONQUE. Pieces, trait, roque, prise en passant. Ni compteur des
   50 coups ni numero de coup. Sans importance : l'entrainement n'utilise ni
   l'un ni l'autre.

Les filtres reprennent ceux de label.py, pour que les deux sources restent
melangeables : pas de mat, pas d'echec, meilleur coup ni prise ni promotion,
score borne. S'y ajoute un filtre de profondeur, que notre etiquetage maison ne
pouvait pas offrir.
"""

import argparse
import datetime
import glob
import os
import struct
import sys
import time
from multiprocessing import Process, Queue

import chess
import pyarrow.parquet as pq

# Le repertoire du script n'est pas toujours dans sys.path quand
# multiprocessing relance l'interpreteur par spawn.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import winjob

RECORD_SIZE = 32
_PACK = struct.Struct("<Q16shBBB3x")
_FEN_CODE = {c: i for i, c in enumerate("PNBRQKpnbrqk")}


def pack_from_fen(fen_board, stm, score_white, result, rule50):
    """Meme empaquetage que label.py et datagen.py, directement depuis le FEN."""
    occ = 0
    codes = []
    sq = 56
    for ch in fen_board:
        if ch == "/":
            sq -= 16
        elif ch.isdigit():
            sq += int(ch)
        else:
            occ |= 1 << sq
            codes.append((sq, _FEN_CODE[ch]))
            sq += 1

    codes.sort()
    pcs = bytearray(16)
    for i, (_, code) in enumerate(codes):
        if i & 1:
            pcs[i >> 1] |= code << 4
        else:
            pcs[i >> 1] = code

    return _PACK.pack(occ, bytes(pcs), score_white, stm, result, min(rule50, 255))


def worker(wid, path, groups, args, out_path, queue):
    """Un travailleur ouvre LUI-MEME le fichier et traite ses propres groupes.

    La premiere version transmettait les positions au travers d'un tuyau : sous
    Windows, serialiser 100 000 tuples coute plus cher que le travail lui-meme,
    et les travailleurs restaient a zero pendant que le parent saturait un
    coeur. Ne circulent plus ici qu'un chemin de fichier et des numeros de
    groupe.
    """
    kept = 0
    stats = {"mat": 0, "profondeur": 0, "score": 0, "echec": 0,
             "tactique": 0, "illegale": 0}
    buf = bytearray()
    pf = pq.ParquetFile(path)
    prev_fen = None

    with open(out_path, "wb") as f:
        for g in groups:
            t = pf.read_row_group(g, columns=["fen", "line", "depth", "cp", "mate"])
            fens = t.column("fen").to_pylist()
            lines = t.column("line").to_pylist()
            deps = t.column("depth").to_pylist()
            cps = t.column("cp").to_pylist()
            mts = t.column("mate").to_pylist()

            for i in range(len(fens)):
                # MultiPV : seule la premiere ligne d'une suite porte
                # l'evaluation de la position.
                if fens[i] == prev_fen:
                    continue
                prev_fen = fens[i]

                # Un mat certain N'EST PAS a jeter. Les mats se concentrent
                # dans les finales : les ecarter revient a supprimer les
                # finales DECISIVES et a ne garder que les nulles. Un reseau
                # entraine ainsi apprend que les finales sont nulles -- mesure
                # a -301 Elo, et invisible sur toutes les statistiques du lot.
                # On les convertit donc en score de gain certain.
                if mts[i] is not None:
                    m = int(mts[i])
                    if m == 0:
                        continue
                    cp = args.mate_score if m > 0 else -args.mate_score
                    stats["mat"] += 1          # comptes, pas ecartes
                else:
                    # La profondeur ne s'applique qu'aux scores en centipions :
                    # un mat force est certain quelle que soit la profondeur.
                    if deps[i] < args.min_depth:
                        stats["profondeur"] += 1
                        continue
                    cp = int(cps[i])
                    if abs(cp) > args.max_score:
                        stats["score"] += 1
                        continue

                try:
                    board = chess.Board(fens[i])
                except ValueError:
                    continue

                # Le jeu contient des positions IMPOSSIBLES -- jusqu'a 35
                # pieces, des rois en double : ce sont des positions posees a
                # la main par des utilisateurs sur l'echiquier d'analyse.
                # Elles ne planteraient pas l'entrainement, elles
                # apprendraient au reseau des echecs qui n'existent pas.
                if not board.is_valid():
                    stats["illegale"] += 1
                    continue

                if board.is_check():
                    stats["echec"] += 1
                    continue

                if args.quiet_only:
                    first = lines[i].split(" ", 1)[0] if lines[i] else ""
                    if not first:
                        continue
                    try:
                        mv = chess.Move.from_uci(first)
                    except ValueError:
                        continue
                    if mv not in board.legal_moves:
                        continue
                    if board.is_capture(mv) or mv.promotion is not None:
                        stats["tactique"] += 1
                        continue

                parts = fens[i].split()
                buf += pack_from_fen(parts[0],
                                     0 if parts[1] == "w" else 1,
                                     cp,       # deja du point de vue des blancs
                                     1,        # resultat inconnu, non utilise
                                     0)
                kept += 1

                if len(buf) >= (1 << 21):
                    f.write(buf)
                    buf.clear()
                    queue.put(("progress", wid, kept, stats))

        if buf:
            f.write(buf)

    queue.put(("done", wid, kept, stats))


def main():
    winjob.install()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=r"C:\Onechess\data\lichess_eval\*.parquet")
    ap.add_argument("--out", default=r"C:\Onechess\data\nnue_lichess")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-depth", type=int, default=20)
    ap.add_argument("--max-score", type=int, default=3000)
    ap.add_argument("--mate-score", type=int, default=2900,
                    help="valeur donnee a un mat force")
    ap.add_argument("--quiet-only", type=int, default=1)
    ap.add_argument("--limit-groups", type=int, default=0,
                    help="0 = tous les groupes de lignes")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    existing = sum(os.path.getsize(p) for p in glob.glob(os.path.join(args.out, "*.bin")))
    if existing:
        print(f"{existing // RECORD_SIZE:,} positions deja presentes, conservees")

    tag = datetime.datetime.now().strftime("%m%d_%H%M")
    files = sorted(glob.glob(args.parquet))
    if not files:
        raise SystemExit(f"aucun fichier ne correspond a {args.parquet}")

    print(f"{len(files)} fichier(s)   profondeur mini {args.min_depth}   "
          f"{args.workers} travailleurs")

    t0 = time.time()
    total_kept = 0
    agg = {"mat": 0, "profondeur": 0, "score": 0, "echec": 0,
           "tactique": 0, "illegale": 0}

    for path in files:
        pf = pq.ParquetFile(path)
        n_groups = pf.metadata.num_row_groups
        if args.limit_groups:
            n_groups = min(n_groups, args.limit_groups)

        # Chaque travailleur recoit des NUMEROS de groupe, pas des donnees. Le
        # decoupage par blocs contigus limite les doublons aux frontieres : une
        # suite de FEN a cheval sur deux blocs peut donner deux enregistrements
        # au lieu d'un, soit moins d'un pour cent mille.
        per = (n_groups + args.workers - 1) // args.workers
        q = Queue()
        procs = []
        base = os.path.basename(path).replace(".parquet", "")
        for w in range(args.workers):
            groups = list(range(w * per, min((w + 1) * per, n_groups)))
            if not groups:
                continue
            out = os.path.join(args.out, f"lic_{tag}_{base}_{w}.bin")
            proc = Process(target=worker, args=(w, path, groups, args, out, q))
            proc.start()
            procs.append(proc)

        done, state = 0, {}
        last = time.time()
        while done < len(procs):
            kind, wid, kept, st = q.get()
            state[wid] = (kept, st)
            if kind == "done":
                done += 1
            if time.time() - last > 15 or kind == "done":
                last = time.time()
                cur = total_kept + sum(k for k, _ in state.values())
                dt = time.time() - t0
                print(f"  {cur:>12,} gardees   {cur / max(dt, 1e-9):7.0f}/s   "
                      f"{done}/{len(procs)} travailleurs finis", flush=True)

        for proc in procs:
            proc.join()
        for kept, st in state.values():
            total_kept += kept
            for k in agg:
                agg[k] += st[k]

    dt = time.time() - t0
    print()
    print(f"termine : {total_kept:,} positions en {dt / 60:.1f} min "
          f"({total_kept / dt:.0f}/s, {total_kept * RECORD_SIZE / 1e9:.2f} Go)")
    print("comptes par motif (les mats sont GARDES, convertis en gain certain) :")
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"   {k:12s} {v:>12,}")


if __name__ == "__main__":
    main()
