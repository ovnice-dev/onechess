#!/usr/bin/env python3
"""Arene de test SPRT pour Onechess.

Fait jouer deux moteurs UCI l'un contre l'autre en parallele et decide, le plus
tot possible, si le premier est meilleur que le second.

Deux points de methode :

  - Les parties sont jouees par PAIRES sur la meme ouverture, couleurs
    inversees. Cela retire du bruit la difficulte propre a l'ouverture, qui est
    la premiere source de variance d'un match.

  - Les statistiques sont pentanomiales : on compte le resultat de la paire
    (0, 0.5, 1, 1.5, 2) et non celui de chaque partie. A nombre de parties egal
    la barre d'erreur est environ 20 % plus serree qu'en trinomial.

Exemple :

    python arena.py --e1 bin/onechess.exe --e2 bin/onechess-base.exe \\
        --tc 8+0.08 --games 2000 --concurrency 6 \\
        --openings ../data/openings.epd --sprt 0 5
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass, field

import chess
import chess.engine

# Le repertoire du script n'est pas toujours dans sys.path quand
# multiprocessing relance l'interpreteur par spawn.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import winjob

# --------------------------------------------------------------------------
# Statistiques
# --------------------------------------------------------------------------

PAIR_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


def elo_to_score(elo):
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score):
    if score <= 0.0:
        return -900.0
    if score >= 1.0:
        return 900.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def pentanomial_stats(counts):
    """Rend (mu, variance de la moyenne, nombre de paires)."""
    n = sum(counts)
    if n == 0:
        return 0.5, 0.0, 0
    probs = [c / n for c in counts]
    mu = sum(p * v for p, v in zip(probs, PAIR_VALUES))
    var = sum(p * (v - mu) ** 2 for p, v in zip(probs, PAIR_VALUES))
    return mu, var / n, n


def llr_pentanomial(counts, elo0, elo1):
    """Rapport de vraisemblance generalise (GSPRT) sur les paires."""
    mu, var_mu, n = pentanomial_stats(counts)
    if n < 2 or var_mu <= 0.0:
        return 0.0
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return (s1 - s0) * (2.0 * mu - s0 - s1) / (2.0 * var_mu)


def elo_with_ci(counts, confidence=0.95):
    """Elo estime et demi-intervalle de confiance, en Elo."""
    mu, var_mu, n = pentanomial_stats(counts)
    if n < 2 or var_mu <= 0.0:
        return 0.0, 0.0
    z = 1.959963985 if confidence == 0.95 else 2.575829304
    lo = max(1e-9, mu - z * math.sqrt(var_mu))
    hi = min(1 - 1e-9, mu + z * math.sqrt(var_mu))
    elo = score_to_elo(mu)
    return elo, (score_to_elo(hi) - score_to_elo(lo)) / 2.0


# --------------------------------------------------------------------------
# Un travailleur : deux moteurs persistants, une partie a la fois
# --------------------------------------------------------------------------

WORKER = {}


@dataclass
class EngineSpec:
    path: str
    name: str
    options: dict = field(default_factory=dict)


def _open_engine(spec, hash_mb, threads):
    # Le chemin est resolu ici : un travailleur lance par spawn n'herite pas
    # forcement du repertoire courant du parent.
    eng = chess.engine.SimpleEngine.popen_uci(spec.path, timeout=30.0)
    opts = {}
    if "Hash" in eng.options:
        opts["Hash"] = hash_mb
    if "Threads" in eng.options:
        opts["Threads"] = threads
    for k, v in spec.options.items():
        if k in eng.options:
            opts[k] = v
        else:
            print(f"[avertissement] {spec.name} ne connait pas l'option {k}", file=sys.stderr)
    if opts:
        eng.configure(opts)
    return eng


def worker_init(spec1, spec2, hash_mb, threads, cfg):
    WORKER["e1"] = _open_engine(spec1, hash_mb, threads)
    WORKER["e2"] = _open_engine(spec2, hash_mb, threads)
    WORKER["spec1"], WORKER["spec2"] = spec1, spec2
    WORKER["hash"], WORKER["threads"] = hash_mb, threads
    WORKER["cfg"] = cfg


def restart(which):
    """Relance un moteur mort. Sans cela, une seule mort ferait echouer toutes
    les parties suivantes du travailleur, et 1 plantage se compterait 50 fois."""
    try:
        WORKER[which].close()
    except Exception:
        pass
    spec = WORKER["spec1"] if which == "e1" else WORKER["spec2"]
    WORKER[which] = _open_engine(spec, WORKER["hash"], WORKER["threads"])


def worker_close():
    for k in ("e1", "e2"):
        if k in WORKER:
            try:
                WORKER[k].quit()
            except Exception:
                pass


def play_game(white, black, opening_fen, cfg, game_id):
    """Rend le score des blancs : 1.0, 0.5 ou 0.0, plus la raison."""
    board = chess.Board(opening_fen)

    # cfg porte les reglages des DEUX moteurs ; white_cfg dit lequel joue blanc
    cw = cfg["w"]
    cb = cfg["b"]
    clocks = {chess.WHITE: cw["base"], chess.BLACK: cb["base"]}
    incs = {chess.WHITE: cw["inc"], chess.BLACK: cb["inc"]}

    # Adjudication : compteurs de coups consecutifs sous seuil
    draw_run = 0
    win_run = 0
    win_side = None

    while True:
        if board.is_game_over(claim_draw=True):
            res = board.result(claim_draw=True)
            return (1.0 if res == "1-0" else 0.0 if res == "0-1" else 0.5), "regle"

        if board.fullmove_number > cfg["max_moves"]:
            return 0.5, "limite de coups"

        stm = board.turn
        eng = white if stm == chess.WHITE else black

        side = cw if stm == chess.WHITE else cb
        inc  = incs[stm]

        if side["nodes"]:
            # A noeuds fixes il n'y a plus d'horloge : ni perte au temps, ni
            # latence de protocole. C'est le seul etalonnage propre face a un
            # moteur beaucoup plus fort, ou les cadences tres courtes ne
            # mesurent plus que le surcout des tuyaux.
            limit = chess.engine.Limit(nodes=side["nodes"])
        elif side["depth"]:
            limit = chess.engine.Limit(depth=side["depth"])
        else:
            limit = chess.engine.Limit(
                white_clock=clocks[chess.WHITE],
                black_clock=clocks[chess.BLACK],
                white_inc=incs[chess.WHITE],
                black_inc=incs[chess.BLACK],
            )

        t0 = time.perf_counter()
        try:
            result = eng.play(board, limit, game=game_id,
                              info=chess.engine.INFO_SCORE, ponder=False)
        except chess.engine.EngineTerminatedError:
            # Le camp au trait est celui dont le moteur vient de mourir :
            # play_pair s'en sert pour savoir lequel relancer.
            return (0.0 if stm == chess.WHITE else 1.0), "moteur mort"
        except Exception:
            return (0.0 if stm == chess.WHITE else 1.0), "erreur moteur"
        elapsed = time.perf_counter() - t0

        if not side["depth"] and not side["nodes"]:
            clocks[stm] -= elapsed
            if clocks[stm] < -cfg["tolerance"]:
                return (0.0 if stm == chess.WHITE else 1.0), "perte au temps"
            clocks[stm] = max(0.0, clocks[stm]) + inc

        move = result.move
        if move is None or move not in board.legal_moves:
            return (0.0 if stm == chess.WHITE else 1.0), "coup illegal"

        # --------------------------------------------------- adjudication
        cp = None
        if result.info and "score" in result.info:
            sc = result.info["score"].pov(chess.WHITE)
            if sc.is_mate():
                cp = 10000 if sc.mate() > 0 else -10000
            else:
                cp = sc.score()

        if cp is not None and cfg["win_cp"]:
            if abs(cp) >= cfg["win_cp"]:
                side = chess.WHITE if cp > 0 else chess.BLACK
                win_run = win_run + 1 if side == win_side else 1
                win_side = side
                if win_run >= cfg["win_moves"] * 2:
                    return (1.0 if win_side == chess.WHITE else 0.0), "adjuge gagne"
            else:
                win_run = 0
                win_side = None

        if cp is not None and cfg["draw_cp"] is not None:
            if board.fullmove_number >= cfg["draw_after"] and abs(cp) <= cfg["draw_cp"]:
                draw_run += 1
                if draw_run >= cfg["draw_moves"] * 2:
                    return 0.5, "adjuge nulle"
            else:
                draw_run = 0

        board.push(move)


def play_pair(task):
    """Joue une ouverture deux fois, couleurs inversees. Rend (s1_a, s1_b)."""
    idx, fen = task
    e1, e2, cfg = WORKER["e1"], WORKER["e2"], WORKER["cfg"]

    cfg_a = dict(cfg, w=cfg["c1"], b=cfg["c2"])
    cfg_b = dict(cfg, w=cfg["c2"], b=cfg["c1"])

    sa, ra = play_game(e1, e2, fen, cfg_a, (idx, 0))   # e1 avec les blancs
    if "mort" in ra:
        # e1 jouait blanc : s'il a perdu, c'est lui qui est mort
        restart("e1" if sa == 0.0 else "e2")
        e1, e2 = WORKER["e1"], WORKER["e2"]
        ra += " (e1)" if sa == 0.0 else " (e2)"

    sb, rb = play_game(e2, e1, fen, cfg_b, (idx, 1))   # e1 avec les noirs
    if "mort" in rb:
        # e2 jouait blanc : s'il a perdu, c'est lui qui est mort
        restart("e2" if sb == 0.0 else "e1")
        rb += " (e2)" if sb == 0.0 else " (e1)"

    return idx, sa, 1.0 - sb, (ra, rb)


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------


def parse_tc(s):
    if "+" in s:
        base, inc = s.split("+")
        return float(base), float(inc)
    return float(s), 0.0


def parse_options(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"option mal formee: {it} (attendu nom=valeur)")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_openings(path, count, seed):
    fens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Un EPD peut porter des annotations apres le champ de roque
            parts = line.split()
            if len(parts) >= 4:
                fen = " ".join(parts[:4])
                if len(parts) >= 6 and parts[4].isdigit() and parts[5].isdigit():
                    fen = " ".join(parts[:6])
                else:
                    fen += " 0 1"
                fens.append(fen)
    if not fens:
        raise SystemExit(f"aucune ouverture lue dans {path}")
    rng = random.Random(seed)
    rng.shuffle(fens)
    if count and count < len(fens):
        fens = fens[:count]
    return fens


def main():
    winjob.install()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1", required=True, help="moteur teste")
    ap.add_argument("--e2", required=True, help="moteur de reference")
    ap.add_argument("--name1", default=None)
    ap.add_argument("--name2", default=None)
    ap.add_argument("--opt1", action="append", help="option UCI pour e1, nom=valeur")
    ap.add_argument("--opt2", action="append", help="option UCI pour e2, nom=valeur")
    ap.add_argument("--tc", default="8+0.08", help="cadence, secondes+increment")
    ap.add_argument("--tc2", default=None, help="cadence de e2 si differente")
    ap.add_argument("--depth1", type=int, default=0, help="profondeur fixe pour e1")
    ap.add_argument("--depth2", type=int, default=0, help="profondeur fixe pour e2")
    ap.add_argument("--nodes1", type=int, default=0, help="noeuds fixes pour e1")
    ap.add_argument("--nodes2", type=int, default=0, help="noeuds fixes pour e2")
    ap.add_argument("--games", type=int, default=2000, help="nombre maximal de parties")
    ap.add_argument("--state", default=None,
                    help="fichier d'etat : un match interrompu reprend son compte "
                         "au lieu de repartir de zero")
    ap.add_argument("--concurrency", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--openings", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sprt", nargs="*", type=float, default=None,
                    metavar="X", help="elo0 elo1 [alpha beta]")
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--win-cp", type=int, default=1000, help="0 pour desactiver")
    ap.add_argument("--win-moves", type=int, default=5)
    ap.add_argument("--draw-cp", type=int, default=8, help="-1 pour desactiver")
    ap.add_argument("--draw-moves", type=int, default=10)
    ap.add_argument("--draw-after", type=int, default=40)
    ap.add_argument("--tolerance", type=float, default=0.5,
                    help="marge de depassement du temps, en secondes")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    base, inc = parse_tc(args.tc)
    base2, inc2 = parse_tc(args.tc2) if args.tc2 else (base, inc)

    e1_path = os.path.abspath(args.e1)
    e2_path = os.path.abspath(args.e2)
    for p in (e1_path, e2_path):
        if not os.path.isfile(p):
            raise SystemExit(f"moteur introuvable : {p}")

    spec1 = EngineSpec(e1_path, args.name1 or os.path.basename(args.e1), parse_options(args.opt1))
    spec2 = EngineSpec(e2_path, args.name2 or os.path.basename(args.e2), parse_options(args.opt2))

    cfg = {
        "c1": {"base": base, "inc": inc, "depth": args.depth1, "nodes": args.nodes1},
        "c2": {"base": base2, "inc": inc2, "depth": args.depth2, "nodes": args.nodes2},
        "max_moves": args.max_moves,
        "win_cp": args.win_cp if args.win_cp > 0 else 0,
        "win_moves": args.win_moves,
        "draw_cp": args.draw_cp if args.draw_cp >= 0 else None,
        "draw_moves": args.draw_moves,
        "draw_after": args.draw_after,
        "tolerance": args.tolerance,
    }

    n_pairs = args.games // 2
    if args.openings:
        fens = load_openings(args.openings, n_pairs, args.seed)
    else:
        fens = [chess.STARTING_FEN]

    tasks = [(i, fens[i % len(fens)]) for i in range(n_pairs)]

    elo0 = elo1 = None
    alpha = beta = 0.05
    if args.sprt is not None:
        if len(args.sprt) >= 2:
            elo0, elo1 = args.sprt[0], args.sprt[1]
        else:
            elo0, elo1 = 0.0, 5.0
        if len(args.sprt) >= 4:
            alpha, beta = args.sprt[2], args.sprt[3]
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)

    counts = [0, 0, 0, 0, 0]   # paires valant 0, 0.5, 1, 1.5, 2
    reasons = {}
    w = d = l = 0
    t_start = time.time()

    # ------------------------------------------------ reprise d'un match
    # Un SPRT interrompu repartait de zero : le 10-09, 2414 parties ont ete
    # perdues ainsi. L'etat est desormais ecrit apres chaque paire, et une
    # relance avec le meme --state reprend exactement la ou elle s'etait
    # arretee.
    #
    # La signature interdit de melanger des parties incompatibles : si un
    # moteur, une option, une cadence ou les ouvertures ont change, le fichier
    # est refuse. Seuls le parallelisme et l'affichage peuvent differer -- ils
    # ne changent pas le resultat des parties.
    signature = {k: v for k, v in sorted(vars(args).items())
                 if k not in ("concurrency", "quiet", "state")}
    done = set()
    if args.state and os.path.exists(args.state):
        with open(args.state, encoding="utf-8") as fh:
            st = json.load(fh)
        if st.get("signature") != signature:
            raise SystemExit(f"{args.state} decrit un autre match : reprise refusee "
                             f"(supprimer le fichier pour repartir de zero)")
        counts = st["counts"]
        reasons = st["reasons"]
        w, d, l = st["w"], st["d"], st["l"]
        done = set(st["done"])
        tasks = [t for t in tasks if t[0] not in done]
        print(f"reprise : {len(done)} paires deja jouees, {len(tasks)} restantes")

    def save_state():
        if not args.state:
            return
        tmp = args.state + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"signature": signature, "counts": counts, "reasons": reasons,
                       "w": w, "d": d, "l": l, "done": sorted(done)}, fh)
        os.replace(tmp, args.state)   # atomique : jamais d'etat a moitie ecrit

    print(f"{spec1.name}  contre  {spec2.name}")
    d1 = (f"{args.nodes1}n" if args.nodes1 else
          f"d{args.depth1}" if args.depth1 else f"{base}+{inc}s")
    d2 = (f"{args.nodes2}n" if args.nodes2 else
          f"d{args.depth2}" if args.depth2 else f"{base2}+{inc2}s")
    print(f"cadence  e1 {d1}   e2 {d2}   hash {args.hash} Mo   "
          f"threads {args.threads}   parallelisme {args.concurrency}")
    if elo0 is not None:
        print(f"SPRT  H0: {elo0:+.1f} Elo   H1: {elo1:+.1f} Elo   "
              f"alpha={alpha} beta={beta}   bornes [{lower:+.2f}, {upper:+.2f}]")
    if args.opt1:
        print(f"options e1 : {spec1.options}")
    if args.opt2:
        print(f"options e2 : {spec2.options}")
    print("-" * 78, flush=True)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=args.concurrency,
        initializer=worker_init,
        initargs=(spec1, spec2, args.hash, args.threads, cfg),
    )

    verdict = "non concluant"
    try:
        for _idx, sa, sb, rs in pool.imap_unordered(play_pair, tasks, chunksize=1):
            for r in rs:
                reasons[r] = reasons.get(r, 0) + 1
            pair = sa + sb
            counts[int(round(pair * 2))] += 1

            for s in (sa, sb):
                if s == 1.0:
                    w += 1
                elif s == 0.0:
                    l += 1
                else:
                    d += 1

            done.add(_idx)
            save_state()

            n_games = w + d + l
            elo, ci = elo_with_ci(counts)

            line = (f"{n_games:5d} parties  +{w} ={d} -{l}   "
                    f"Elo {elo:+7.1f} +/- {ci:5.1f}")

            if elo0 is not None:
                llr = llr_pentanomial(counts, elo0, elo1)
                line += f"   LLR {llr:+6.2f} [{lower:+.2f},{upper:+.2f}]"

                if llr >= upper:
                    verdict = "H1 ACCEPTEE (le changement gagne)"
                    print(line, flush=True)
                    break
                if llr <= lower:
                    verdict = "H0 ACCEPTEE (le changement ne gagne pas)"
                    print(line, flush=True)
                    break

            if not args.quiet:
                print(line, flush=True)
    except KeyboardInterrupt:
        verdict = "interrompu"
    finally:
        pool.terminate()
        pool.join()

    elapsed = time.time() - t_start
    n_games = w + d + l
    elo, ci = elo_with_ci(counts)

    print("-" * 78)
    print(f"{n_games} parties en {elapsed:.0f} s   +{w} ={d} -{l}")
    print(f"paires  0:{counts[0]}  0.5:{counts[1]}  1:{counts[2]}  "
          f"1.5:{counts[3]}  2:{counts[4]}")
    print(f"Elo {elo:+.1f} +/- {ci:.1f}   ({spec1.name} vs {spec2.name})")
    if reasons:
        detail = "  ".join(f"{k}:{v}" for k, v in sorted(reasons.items(),
                                                         key=lambda kv: -kv[1]))
        print(f"fins de partie  {detail}")
    if elo0 is not None:
        print(f"LLR {llr_pentanomial(counts, elo0, elo1):+.2f}   verdict : {verdict}")

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
