"""Gestion du temps en cadence « N coups / temps » (celle du CCRL 40/15), coup par coup.

Onechess contre lui-meme, pendule reelle avec movestogo ; on releve pour chaque
controle de temps la part du temps depensee et la pendule la plus basse, les
pertes au temps, et le temps moyen par coup selon la position dans le controle.

    python tools/tm_trace.py --engine engine/bin/onechess.exe --tc 40/20 --parties 8 --par 4
"""
import argparse
import concurrent.futures as cf
import random
import time

import chess
import chess.engine


def partie(args, fen, k):
    n, base = args.tc.split("/")
    n, base = int(n), float(base)
    e = chess.engine.SimpleEngine.popen_uci(args.engine)
    e.configure({"Hash": 64, "Threads": 1, **({"EvalFile": args.net} if args.net else {})})
    b = chess.Board(fen)
    clocks = {chess.WHITE: base, chess.BLACK: base}
    joues = {chess.WHITE: 0, chess.BLACK: 0}
    traces = []          # (camp, coup dans le controle 1..n, temps depense, pendule apres)
    perte = None
    while not b.is_game_over(claim_draw=True) and b.fullmove_number < 200:
        stm = b.turn
        lim = chess.engine.Limit(white_clock=clocks[chess.WHITE], black_clock=clocks[chess.BLACK],
                                 remaining_moves=n - joues[stm] % n)
        t0 = time.perf_counter()
        r = e.play(b, lim)
        dt = time.perf_counter() - t0
        clocks[stm] -= dt
        rang = joues[stm] % n + 1
        traces.append((stm, rang, dt, clocks[stm]))
        if clocks[stm] < 0:
            perte = (stm, joues[stm] + 1, clocks[stm])
            break
        joues[stm] += 1
        if joues[stm] % n == 0:
            clocks[stm] += base
        b.push(r.move)
    e.quit()
    return traces, perte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--net", default="")
    ap.add_argument("--tc", default="40/20")
    ap.add_argument("--parties", type=int, default=8)
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--openings", default="data/uho_085.epd")
    a = ap.parse_args()
    n, base = int(a.tc.split("/")[0]), float(a.tc.split("/")[1])
    fens = [l.split(";")[0].strip() for l in open(a.openings) if l.strip()]
    random.seed(1)
    choix = random.sample(fens, a.parties)
    with cf.ThreadPoolExecutor(a.par) as ex:
        res = list(ex.map(lambda x: partie(a, x[1], x[0]), enumerate(choix)))
    pertes = [p for _, p in res if p]
    # par controle complet : temps depense / temps disponible, pendule minimale
    parts, mins, par_rang = [], [], {}
    for tr, _ in res:
        for camp in (chess.WHITE, chess.BLACK):
            t = [x for x in tr if x[0] == camp]
            for i in range(0, len(t) - n + 1, n):
                bloc = t[i:i + n]
                if len(bloc) == n:
                    parts.append(sum(x[2] for x in bloc) / base)
                    mins.append(min(x[3] for x in bloc))
            for x in t:
                par_rang.setdefault(x[1], []).append(x[2])
    print(f"cadence {a.tc} : {a.parties} parties, {len(pertes)} perte(s) au temps {pertes}")
    if parts:
        parts.sort(); mins.sort()
        print(f"controles complets : {len(parts)} ; part du temps depensee : min {parts[0]:.2f}, "
              f"mediane {parts[len(parts)//2]:.2f}, max {parts[-1]:.2f}")
        print(f"pendule la plus basse avant le controle : min {mins[0]:.2f} s, mediane {mins[len(mins)//2]:.2f} s "
              f"(sur {base:.0f} s)")
    ligne = []
    for r in (1, 5, 10, 20, 30, 35, 38, 39, 40):
        if r in par_rang:
            v = par_rang[r]
            ligne.append(f"{r}:{sum(v)/len(v):.2f}s")
    print("temps moyen par coup selon son rang dans le controle :", " ".join(ligne))


if __name__ == "__main__":
    main()
