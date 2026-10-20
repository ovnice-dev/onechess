#!/usr/bin/env bash
# Nos donnees contre celles de Lichess, a VOLUME EGAL.
#
# Comparer les erreurs de validation n'aurait aucun sens : chaque reseau serait
# valide sur sa propre distribution, et deux distributions n'ont pas la meme
# difficulte intrinseque. Le chiffre le plus bas ne designerait pas le meilleur
# reseau, seulement la distribution la plus facile.
#
# On entraine donc les deux a l'identique -- meme architecture, meme graine,
# meme nombre de positions, meme nombre d'epoques -- et on les fait JOUER l'un
# contre l'autre. C'est le seul arbitre qui ne depend pas de la distribution.
set -u
cd /c/Onechess
LOG=data/duel_sources.log
N=7142775

echo "=== Lichess, $N positions  ($(date +%H:%M)) ===" | tee -a "$LOG"
python -u tools/train.py --data 'data/nnue_lichess/*.bin' --cache data/cache_lic.npz \
    --seed 0 --epochs 40 --hidden 512 --buckets plain --mirror 0 \
    --out-buckets 4 --activation crelu --out engine/bin/E_lichess.nnue 2>&1 \
    | stdbuf -oL grep -E "positions lues|RESULTAT|reseau ecrit|Error|Traceback" | tee -a "$LOG"

echo "=== nos donnees, $N positions  ($(date +%H:%M)) ===" | tee -a "$LOG"
python -u tools/train.py --data 'data/nnue_big/*.bin' --cache data/cache_full.npz \
    --use $N --seed 0 --epochs 40 --hidden 512 --buckets plain --mirror 0 \
    --out-buckets 4 --activation crelu --out engine/bin/E_nous.nnue 2>&1 \
    | stdbuf -oL grep -E "positions lues|cache lu|RESULTAT|reseau ecrit|Error|Traceback" | tee -a "$LOG"

echo "### termine $(date +%H:%M)" | tee -a "$LOG"
