#!/usr/bin/env bash
# Les deux variantes qui manquaient au balayage du 06-09.
# Le cache est deja construit : chaque tour attaque directement l'entrainement.
set -u
cd /c/Onechess

CACHE=data/cache_full.npz
LOG=data/sweep_A.log
DATA='data/nnue_big/*.bin'
COMMON="--cache $CACHE --epochs 40 --seed 0"

run() {
    local nom="$1"; shift
    echo "=== $nom  ($(date +%H:%M)) ===" | tee -a "$LOG"
    # stdbuf force le vidage ligne par ligne : sans lui le filtre retient tout
    # jusqu'a la fin du tour, et on ne voit aucune progression.
    python -u tools/train.py --data "$DATA" $COMMON "$@" \
        --out "engine/bin/B_${nom}.nnue" 2>&1 \
        | stdbuf -oL grep -E "positions|epoque|RESULTAT|reseau ecrit|Error|Traceback" \
        | tee -a "$LOG"
    echo | tee -a "$LOG"
}

# 4. Buckets de sortie : 4 etait l'optimum a 1,5 M seulement.
run ob8    --hidden 512  --buckets plain --mirror 0 --out-buckets 8 --activation crelu

# 5. LA variante decisive : le reseau est desormais limite par sa capacite,
#    plus par les donnees. Peut deborder des 6 Go de la carte.
run hl1024 --hidden 1024 --buckets plain --mirror 0 --out-buckets 4 --activation crelu

echo "### balayage termine le $(date '+%Y-%m-%d %H:%M')" | tee -a "$LOG"
