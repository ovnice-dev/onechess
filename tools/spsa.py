#!/usr/bin/env python3
"""Reglage simultane des constantes de recherche par SPSA.

Un test SPRT ne mesure qu'une constante a la fois et demande des milliers de
parties par constante. Avec une soixantaine de constantes reglables, c'est hors
de portee d'une machine de bureau.

SPSA (Simultaneous Perturbation Stochastic Approximation) estime le gradient de
la force de jeu par rapport a TOUTES les constantes a la fois : a chaque
iteration on tire un signe au hasard pour chacune, on fait jouer la version
"tous les +" contre la version "tous les -", et le resultat de la partie donne
une estimation bruitee mais non biaisee du gradient. Le meme jeu de parties
sert donc a regler tous les parametres.

Les parties sont jouees par PAIRES sur la meme ouverture, couleurs inversees,
pour retirer du bruit la difficulte propre a l'ouverture.

Exemple :

    python spsa.py --iters 4000 --tc 1+0.01 --workers 6 \\
        --params LmrBase,20,150 LmrDivisor,120,400 RfpMargin,40,200
"""

import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import Process, Queue

import chess
import chess.engine

# Le repertoire du script n'est pas toujours dans sys.path quand
# multiprocessing relance l'interpreteur par spawn.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import winjob

# Constantes candidates par defaut : celles dont on sait qu'elles pesent, et
# qui n'ont jamais ete balayees.
DEFAULT_PARAMS = [
    # nom,               min,  max,  valeur de depart
    ("LmrBase",           20,  150,   77),
    ("LmrDivisor",       120,  400,  233),
    ("LmrHistDiv",      2000, 30000, 9000),
    ("RfpMargin",         40,  200,   78),
    ("RfpMaxDepth",        3,   15,    8),
    ("NmpBaseReduction",   2,    6,    4),
    ("NmpEvalDiv",        50,  600,  200),
    ("FutilityMargin",    50,  250,  110),
    ("HistPruneMargin", 1000, 12000, 3500),
    ("LmpBase",            1,    8,    3),
    ("SeeCaptureMargin",  50,  400,  160),
    ("SeeQuietMargin",     5,  120,   25),
    ("ProbcutMargin",     80,  350,  180),
    ("SingularBetaFactor", 8,   64,   24),
    ("StatBonusMul",      50,  400,  190),
    ("StatMalusMul",     100,  900,  550),
    ("QsFutilityMargin",  50,  400,  130),
    ("AspirationDelta",    4,   50,   10),
]


# --------------------------------------------------------------------------
# Travailleur : deux moteurs, une paire de parties par tache
# --------------------------------------------------------------------------

W = {}


def worker_init(engine_path, net, hash_mb, cfg):
    W["a"] = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=30)
    W["b"] = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=30)
    base = {"Hash": hash_mb, "Threads": 1}
    if net:
        base["EvalFile"] = net
    for e in (W["a"], W["b"]):
        e.configure(base)
    W["cfg"] = cfg


def play_game(white, black, fen, cfg, gid):
    board = chess.Board(fen)
    clocks = {chess.WHITE: cfg["base"], chess.BLACK: cfg["base"]}
    inc = cfg["inc"]
    run, side = 0, None

    while True:
        if board.is_game_over(claim_draw=True):
            r = board.result(claim_draw=True)
            return 1.0 if r == "1-0" else 0.0 if r == "0-1" else 0.5
        if board.fullmove_number > cfg["max_moves"]:
            return 0.5

        stm = board.turn
        eng = white if stm == chess.WHITE else black

        # A noeuds fixes il n'y a pas d'horloge du tout : les deux moteurs
        # cherchent exactement le meme nombre de noeuds quelle que soit la
        # charge de la machine. Le resultat ne depend que de la qualite des
        # coups, donc le reglage peut tourner pendant un entrainement sans
        # etre fausse. La contrepartie est qu'on ne regle plus rien de ce qui
        # touche a la gestion du temps.
        if cfg["nodes"]:
            lim = chess.engine.Limit(nodes=cfg["nodes"])
        else:
            lim = chess.engine.Limit(white_clock=clocks[chess.WHITE],
                                     black_clock=clocks[chess.BLACK],
                                     white_inc=inc, black_inc=inc)

        t0 = time.perf_counter()
        try:
            res = eng.play(board, lim, game=gid, info=chess.engine.INFO_SCORE, ponder=False)
        except Exception:
            return 0.0 if stm == chess.WHITE else 1.0

        if not cfg["nodes"]:
            clocks[stm] -= time.perf_counter() - t0
            if clocks[stm] < -cfg["tolerance"]:
                return 0.0 if stm == chess.WHITE else 1.0
            clocks[stm] = max(0.0, clocks[stm]) + inc

        if res.move is None or res.move not in board.legal_moves:
            return 0.0 if stm == chess.WHITE else 1.0

        # Adjudication : une partie deja decidee n'apporte plus d'information
        cp = None
        if res.info and "score" in res.info:
            sc = res.info["score"].pov(chess.WHITE)
            cp = 10000 if (sc.is_mate() and sc.mate() > 0) else \
                 -10000 if sc.is_mate() else sc.score()
        if cp is not None and abs(cp) >= cfg["win_cp"]:
            s = chess.WHITE if cp > 0 else chess.BLACK
            run = run + 1 if s == side else 1
            side = s
            if run >= cfg["win_plies"]:
                return 1.0 if side == chess.WHITE else 0.0
        else:
            run, side = 0, None

        board.push(res.move)


def run_task(task):
    """task = (idx, fen, options_plus, options_moins). Rend le score de 'plus'."""
    idx, fen, opt_p, opt_m = task
    a, b, cfg = W["a"], W["b"], W["cfg"]

    a.configure(opt_p)
    b.configure(opt_m)

    s1 = play_game(a, b, fen, cfg, (idx, 0))          # plus avec les blancs
    s2 = play_game(b, a, fen, cfg, (idx, 1))          # plus avec les noirs
    return idx, (s1 + (1.0 - s2)) / 2.0


def worker_loop(in_q, out_q, engine_path, net, hash_mb, cfg):
    worker_init(engine_path, net, hash_mb, cfg)
    while True:
        t = in_q.get()
        if t is None:
            break
        out_q.put(run_task(t))
    for k in ("a", "b"):
        try:
            W[k].quit()
        except Exception:
            pass


# --------------------------------------------------------------------------

def parse_params(specs):
    out = []
    for s in specs:
        parts = s.split(",")
        if len(parts) == 3:
            name, lo, hi = parts
            init = None
        elif len(parts) == 4:
            name, lo, hi, init = parts
        else:
            raise SystemExit(f"parametre mal forme : {s} (attendu nom,min,max[,depart])")
        lo, hi = int(lo), int(hi)
        out.append((name.strip(), lo, hi, int(init) if init else (lo + hi) // 2))
    return out


def main():
    winjob.install()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default=r"C:\Onechess\engine\bin\onechess.exe")
    ap.add_argument("--net", default=r"C:/Onechess/engine/bin/onechess.nnue")
    ap.add_argument("--openings", default=r"C:\Onechess\data\openings.epd")
    ap.add_argument("--out", default=r"C:\Onechess\data\spsa.json")
    ap.add_argument("--params", nargs="*", default=None,
                    help="nom,min,max[,depart] ; par defaut la liste integree")
    ap.add_argument("--iters", type=int, default=4000, help="paires de parties")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tc", default="1+0.01")
    ap.add_argument("--nodes", type=int, default=0,
                    help="noeuds fixes par coup ; 0 = jouer a la cadence --tc. "
                         "A noeuds fixes le resultat ne depend plus de la "
                         "charge de la machine, donc le reglage peut tourner "
                         "pendant un entrainement.")
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.602)
    ap.add_argument("--gamma", type=float, default=0.101)
    ap.add_argument("--a-ratio", type=float, default=0.08,
                    help="pas d'apprentissage, en fraction de l'etendue du parametre")
    ap.add_argument("--c-ratio", type=float, default=0.12,
                    help="amplitude de perturbation, en fraction de l'etendue")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    base, inc = (args.tc.split("+") + ["0"])[:2]
    cfg = {"base": float(base), "inc": float(inc), "max_moves": 160,
           "win_cp": 1200, "win_plies": 8, "tolerance": 0.4,
           "nodes": args.nodes}

    if args.nodes:
        print(f"limite : {args.nodes:,} noeuds par coup (aucune horloge)")
    else:
        print(f"limite : cadence {args.tc}")

    params = parse_params(args.params) if args.params else \
        [(n, lo, hi, init) for n, lo, hi, init in DEFAULT_PARAMS]

    theta = [float(init) for _, _, _, init in params]
    names = [n for n, _, _, _ in params]
    lows  = [lo for _, lo, _, _ in params]
    highs = [hi for _, _, hi, _ in params]
    spans = [hi - lo for lo, hi in zip(lows, highs)]

    fens = [l.strip() for l in open(args.openings, encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(fens)

    A = args.iters * 0.1          # amortit les premieres iterations
    a0 = [args.a_ratio * s for s in spans]
    c0 = [max(1.0, args.c_ratio * s) for s in spans]

    limite = f"{args.nodes:,} noeuds" if args.nodes else f"cadence {args.tc}"
    print(f"SPSA : {len(params)} constantes, {args.iters} paires, {limite}, "
          f"{args.workers} travailleurs")
    for n, lo, hi, init in params:
        print(f"   {n:20s} [{lo:6d}, {hi:6d}]  depart {init}")
    print("-" * 78, flush=True)

    in_q, out_q = Queue(maxsize=args.workers * 4), Queue()
    procs = [Process(target=worker_loop,
                     args=(in_q, out_q, args.engine, args.net, args.hash, cfg))
             for _ in range(args.workers)]
    for p in procs:
        p.start()

    pending = {}
    done = 0
    t0 = time.time()
    next_iter = 0

    def submit(k):
        ck = [c / (k + 1) ** args.gamma for c in c0]
        delta = [rng.choice((-1, 1)) for _ in params]
        plus, minus = {}, {}
        for i, n in enumerate(names):
            p = min(highs[i], max(lows[i], theta[i] + ck[i] * delta[i]))
            m = min(highs[i], max(lows[i], theta[i] - ck[i] * delta[i]))
            plus[n], minus[n] = int(round(p)), int(round(m))
        pending[k] = (delta, ck, plus, minus)
        in_q.put((k, fens[k % len(fens)], plus, minus))

    for _ in range(args.workers * 2):
        if next_iter < args.iters:
            submit(next_iter)
            next_iter += 1

    while done < args.iters:
        k, score = out_q.get()
        delta, ck, _, _ = pending.pop(k)
        done += 1

        # Gradient estime : (score - 0.5) porte le signe de l'amelioration
        ak = [a / (k + 1 + A) ** args.alpha for a in a0]
        for i in range(len(theta)):
            theta[i] += ak[i] * (score - 0.5) * 2.0 * delta[i]
            theta[i] = min(highs[i], max(lows[i], theta[i]))

        if next_iter < args.iters:
            submit(next_iter)
            next_iter += 1

        if done % 100 == 0 or done == args.iters:
            dt = time.time() - t0
            print(f"  {done:6d}/{args.iters} paires  {2*done/dt:5.2f} parties/s  "
                  + "  ".join(f"{n}={theta[i]:.0f}" for i, n in enumerate(names[:6])),
                  flush=True)
            json.dump({n: round(theta[i]) for i, n in enumerate(names)},
                      open(args.out, "w"), indent=2)

    for _ in procs:
        in_q.put(None)
    for p in procs:
        p.join(timeout=10)

    print("-" * 78)
    print("valeurs finales :")
    for i, n in enumerate(names):
        print(f"   {n:20s} {params[i][3]:6d}  ->  {round(theta[i]):6d}")
    json.dump({n: round(theta[i]) for i, n in enumerate(names)},
              open(args.out, "w"), indent=2)
    print(f"\necrit dans {args.out}")
    print("Ces valeurs restent a VALIDER par un SPRT contre les valeurs de depart :")
    print("  SPSA indique une direction, il ne prouve pas un gain.")


if __name__ == "__main__":
    main()
