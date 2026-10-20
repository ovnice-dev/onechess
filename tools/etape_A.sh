#!/usr/bin/env bash
# Point d'etape de la phase A : des que la campagne atteint le seuil, entraine
# les architectures candidates sur les donnees FILTREES (lot 2) et compare.
#
# Toutes gardent 4 buckets de sortie, qui sont le seul reglage deja tranche.
set -u
SEUIL="${1:-8000000}"
DATA=/c/Onechess/data/nnue_big
CACHE=/c/Onechess/data/cache_A.npz
LOG=/c/Onechess/data/etape_A_${SEUIL}.log

count() { ls -l "$DATA"/*.bin 2>/dev/null | awk '{s+=$5} END {print int(s/32)+0}'; }

echo "attente de $SEUIL positions filtrees (actuellement $(count))" > "$LOG"
until [ "$(count)" -ge "$SEUIL" ]; do sleep 120; done
echo "seuil atteint a $(date '+%H:%M') : $(count) positions" >> "$LOG"
rm -f "$CACHE"

for cfg in "plain 0 0" "k4 1 1" "k8 1 1"; do
  set -- $cfg
  python /c/Onechess/tools/train.py \
      --data "$DATA/*.bin" --cache "$CACHE" --limit "$SEUIL" \
      --epochs 30 --hidden 512 --buckets "$1" --mirror "$2" --factorizer "$3" \
      --out-buckets 4 --quiet \
      --out "/c/Onechess/engine/bin/A_$1.nnue" 2>&1 \
    | grep -E "^RESULTAT|reseau ecrit" >> "$LOG"
done

echo "termine a $(date '+%H:%M')" >> "$LOG"
cat "$LOG"
