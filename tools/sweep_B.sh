#!/usr/bin/env bash
# Suite du balayage, une fois le bruit d'entrainement connu (< 0,1 cp).
set -u
cd /c/Onechess
LOG=data/sweep_B.log
run() {
    local nom="$1"; shift
    echo "=== $nom  ($(date +%H:%M)) ===" | tee -a "$LOG"
    python -u tools/train.py --data 'data/nnue_big/*.bin' --cache data/cache_full.npz \
        --seed 0 "$@" --out "engine/bin/C_${nom}.nnue" 2>&1 \
        | stdbuf -oL grep -E "RESULTAT|reseau ecrit|Error|Traceback" | tee -a "$LOG"
}

# 1. Les trois reseaux s'arretaient sur leur meilleure epoque : ils
#    s'amelioraient encore. Une seule variable change ici, le nombre d'epoques.
run ep80    --epochs 80 --hidden 512  --buckets plain --mirror 0 --out-buckets 4 --activation crelu

# 2. Les trois gains mesures se cumulent-ils ? Chacun etait petit mais reel.
run combine --epochs 40 --hidden 1024 --buckets plain --mirror 0 --out-buckets 8 --activation screlu

echo "### termine $(date +%H:%M)" | tee -a "$LOG"
