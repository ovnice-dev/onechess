#!/usr/bin/env python3
"""Note CCRL d'Onechess a partir du tournoi d'ancrage (tools/gauntlet.sh).

Chaque adversaire a une note CCRL 40/15 mono-coeur connue (liste du
10 septembre 2026). Un match donne notre ecart d'Elo avec lui, donc une note
pour Onechess ; les notes tirees des differents matchs sont combinees en
moyenne ponderee par l'inverse de leur variance.

Deux incertitudes s'additionnent : celle de nos matchs (pentanomiale, comme
dans arena.py) et celle de la note CCRL de l'adversaire (marges CCRL a 95 %).

Reserve : la cadence du tournoi (10+0.1) est bien plus courte que 40/15, et
les ecarts d'Elo s'etirent aux cadences courtes. Contre des adversaires
proches de notre niveau, la deformation reste petite ; contre un adversaire
lointain, elle ne l'est plus. D'ou le choix d'adversaires proches.

Utilisable a tout moment, y compris pendant le tournoi.
"""

import json
import math
import os
import random

OUT = "data/gauntlet"

# nom du match -> (note CCRL 40/15 mono-coeur, marge a 95 %)
ANCHORS = {
    "Leorik-3.0": (3225, 12),
    "Weiss-2.0": (3264, 10),
    "Avalanche-2.1.0": (3350, 9),
    "Reckless-0.7.0": (3421, 12),
    "RubiChess-20240817": (3569, 7),
}

PAIR_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
Z95 = 1.959963985


def pentanomial(counts):
    """Score moyen et variance de cette moyenne, sur les paires."""
    n = sum(counts)
    mu = sum(c * v for c, v in zip(counts, PAIR_VALUES)) / n
    var = sum(c * (v - mu) ** 2 for c, v in zip(counts, PAIR_VALUES)) / n
    return mu, var / n, n


def score_to_elo(s):
    return -400.0 * math.log10(1.0 / s - 1.0)


def fit_stretch(pts):
    """Ajuste ecart = k (R - A) par moindres carres ponderes.

    pts : (note CCRL de l'adversaire, notre ecart mesure, son ecart-type).

    Aux cadences courtes, les ecarts d'Elo sont plus grands qu'en 40/15 : on
    ecrase davantage les faibles et on perd davantage contre les forts. La
    moyenne simple en depend, selon qu'on a choisi des adversaires plutot
    faibles ou plutot forts. k mesure cet etirement. R est la note a laquelle
    notre ecart s'annule : jouer a egalite ne depend pas de k, et R ne depend
    donc pas de la cadence, du moins tant que l'etirement est le meme de part
    et d'autre.
    """
    w = [1.0 / s ** 2 for _, _, s in pts]
    sw = sum(w)
    ma = sum(wi * a for wi, (a, _, _) in zip(w, pts)) / sw
    md = sum(wi * d for wi, (_, d, _) in zip(w, pts)) / sw
    sxy = sum(wi * (a - ma) * (d - md) for wi, (a, d, _) in zip(w, pts))
    sxx = sum(wi * (a - ma) ** 2 for wi, (a, _, _) in zip(w, pts))
    k = -sxy / sxx
    return (ma + md / k if k > 0 else float("nan")), k


def main():
    os.chdir("/Onechess" if os.path.isdir("/Onechess") else "C:/Onechess")
    rows, num, den = [], 0.0, 0.0
    for name, (anchor, margin) in ANCHORS.items():
        path = os.path.join(OUT, name + ".state")
        if not os.path.exists(path):
            continue
        st = json.load(open(path, encoding="utf-8"))
        mu, var_mu, n = pentanomial(st["counts"])
        if n < 2 or var_mu <= 0.0 or not 0.0 < mu < 1.0:
            continue
        diff = score_to_elo(mu)
        # Methode delta : dElo/dscore = 400 / (ln 10 . s (1 - s)).
        se_match = 400.0 / (math.log(10) * mu * (1.0 - mu)) * math.sqrt(var_mu)
        se = math.hypot(se_match, margin / Z95)
        rating = anchor + diff
        rows.append((name, anchor, 2 * n, st["w"], st["d"], st["l"], mu, diff,
                     Z95 * se_match, rating, Z95 * se, st.get("reasons", {})))
        num += rating / se ** 2
        den += 1.0 / se ** 2

    if not rows:
        print("aucun match termine")
        return

    print(f"\n{'adversaire':<20} {'CCRL':>5} {'parties':>7}  {'+':>4} {'=':>4} {'-':>4}  "
          f"{'score':>6}  {'ecart':>13}  {'note Onechess':>14}")
    for name, anchor, g, w, d, l, mu, diff, e_diff, rating, e_rating, _ in rows:
        print(f"{name:<20} {anchor:>5} {g:>7}  {w:>4} {d:>4} {l:>4}  {100 * mu:5.1f}%  "
              f"{diff:+6.1f} +/-{e_diff:5.1f}  {rating:6.0f} +/-{e_rating:4.0f}")

    est, err = num / den, Z95 / math.sqrt(den)
    print(f"\nMoyenne ponderee (suppose des ecarts identiques a 40/15) : {est:.0f} +/- {err:.0f}  (95 %)")

    if len(rows) >= 3:
        pts = [(r[1], r[7], r[8] / Z95, ANCHORS[r[0]][1] / Z95) for r in rows]
        R, k = fit_stretch([(a, d, s) for a, d, s, _ in pts])
        rng = random.Random(1)
        boot = []
        for _ in range(4000):
            res = fit_stretch([(a + rng.gauss(0, sa), d + rng.gauss(0, s), s) for a, d, s, sa in pts])
            if res[1] > 0.2:
                boot.append(res)
        boot.sort()
        ks = sorted(b[1] for b in boot)
        lo, hi = boot[int(0.025 * len(boot))][0], boot[int(0.975 * len(boot))][0]
        print(f"Point d'equilibre (independant de l'etirement)      : {R:.0f}  [{lo:.0f} ; {hi:.0f}]  (95 %)")
        print(f"  etirement des ecarts a cette cadence : x{k:.2f}  "
              f"[{ks[int(0.025 * len(ks))]:.2f} ; {ks[int(0.975 * len(ks))]:.2f}]")

    losses = sum(r[11].get("perte au temps", 0) for r in rows)
    games = sum(r[2] for r in rows)
    print(f"pertes au temps : {losses} sur {games} parties")


if __name__ == "__main__":
    main()
