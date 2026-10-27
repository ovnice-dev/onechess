#!/usr/bin/env python3
"""Moyenne des SPSA d'un lot GitHub (spsa_lot.yml) -> options UCI.
    gh run download <id> -R ovnice-dev/onechess -D ci_resultats/spsa/<nom>
    python ci/spsa_moy.py ci_resultats/spsa/<nom>
Imprime "Nom=valeur ..." (moyenne arrondie), a passer en opt1 d'un SPRT."""
import glob, json, sys, statistics
fs = sorted(glob.glob(sys.argv[1] + "/**/spsa_m*.json", recursive=True))
if not fs:
    sys.exit("aucun resultat")
res = [json.load(open(f)) for f in fs]
print(f"# {len(res)} SPSA", file=sys.stderr)
for k in res[0]:
    v = [r[k] for r in res]
    print(f"#  {k:20s} moyenne {statistics.mean(v):9.1f}  ecart-type {statistics.pstdev(v):7.1f}", file=sys.stderr)
print(" ".join(f"{k}={round(statistics.mean(r[k] for r in res))}" for k in res[0]))
