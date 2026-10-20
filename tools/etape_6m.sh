#!/usr/bin/env bash
# Attend que la campagne d'etiquetage atteigne un seuil, puis entraine les
# architectures candidates sur ces donnees et compare.
#
# L'interet d'un point intermediaire : savoir si les buckets decollent avant
# d'avoir attendu la campagne entiere.
set -u

SEUIL="${1:-6000000}"
DATA=/c/Onechess/data/nnue_big
CACHE=/c/Onechess/data/cache_big.npz
LOG=/c/Onechess/data/etape_${SEUIL}.log

count() { ls -l "$DATA"/*.bin 2>/dev/null | awk '{s+=$5} END {print int(s/32)+0}'; }

echo "attente de $SEUIL positions (actuellement $(count))" > "$LOG"
until [ "$(count)" -ge "$SEUIL" ]; do sleep 60; done
echo "seuil atteint a $(date '+%H:%M') : $(count) positions" >> "$LOG"

rm -f "$CACHE"

for cfg in "plain 0" "k4 1" "k8 1"; do
  set -- $cfg
  echo "--- buckets=$1 miroir=$2 ---" >> "$LOG"
  python /c/Onechess/tools/train.py \
      --data "$DATA/*.bin" --cache "$CACHE" --limit "$SEUIL" \
      --epochs 30 --hidden 512 --buckets "$1" --mirror "$2" --quiet \
      --out "/c/Onechess/engine/bin/cand_$1_$2.nnue" 2>&1 \
    | grep -E "^RESULTAT|reseau ecrit|entrees," >> "$LOG"
done

echo "termine a $(date '+%H:%M')" >> "$LOG"
cat "$LOG"
