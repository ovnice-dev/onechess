#!/usr/bin/env bash
# Le cumul des seuls gains GRATUITS a l'execution : 80 epoques + 8 buckets de
# sortie. Ni l'un ni l'autre ne coute un noeud par seconde, contrairement a
# screlu et hl1024 qui doivent encore payer leur cout en vitesse.
set -u
cd /c/Onechess
LOG=data/sweep_C.log
echo "=== gratuit : ep80 + ob8  ($(date +%H:%M)) ===" | tee -a "$LOG"
python -u tools/train.py --data 'data/nnue_big/*.bin' --cache data/cache_full.npz \
    --seed 0 --epochs 80 --hidden 512 --buckets plain --mirror 0 \
    --out-buckets 8 --activation crelu --out engine/bin/D_gratuit.nnue 2>&1 \
    | stdbuf -oL grep -E "RESULTAT|reseau ecrit|Error|Traceback" | tee -a "$LOG"
echo "### termine $(date +%H:%M)" | tee -a "$LOG"
