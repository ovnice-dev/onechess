#!/usr/bin/env bash
# Quelle est la dispersion d'un entrainement dont on ne change QUE la graine ?
#
# Les cinq architectures du balayage se tiennent en 0,4 cp. Sans connaitre le
# bruit propre a l'entrainement, ce chiffre ne veut rien dire : il peut etre
# un vrai classement comme un tirage au sort. Deux graines de plus sur
# l'architecture de reference donnent l'etendue, et donc la reponse.
set -u
cd /c/Onechess
LOG=data/seed_noise.log
for s in 1 2; do
    echo "=== plain graine $s  ($(date +%H:%M)) ===" | tee -a "$LOG"
    python -u tools/train.py --data 'data/nnue_big/*.bin' --cache data/cache_full.npz \
        --epochs 40 --seed "$s" --hidden 512 --buckets plain --mirror 0 \
        --out-buckets 4 --activation crelu --out "engine/bin/B_plain_s$s.nnue" 2>&1 \
        | stdbuf -oL grep -E "RESULTAT|reseau ecrit|Error|Traceback" | tee -a "$LOG"
done
echo "### termine $(date +%H:%M)" | tee -a "$LOG"
