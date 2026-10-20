#!/usr/bin/env bash
# Balayage d'architecture sur le jeu de donnees complet.
#
# Toutes les variantes partagent LE MEME cache et LA MEME graine, donc la meme
# decoupe entrainement/validation : les erreurs en centipions sont directement
# comparables entre elles.
#
# Rappel de methode : l'erreur en cp a deja trompe une fois. Elle sert ici a
# ELIMINER, pas a conclure. Ce qui sort en tete passe ensuite par un match.
set -u
cd /c/Onechess

CACHE=data/cache_full.npz
LOG=data/sweep_A.log
DATA='data/nnue_big/*.bin'     # entre apostrophes : c'est train.py qui doit
                               # developper le motif, pas le shell
COMMON="--cache $CACHE --epochs 40 --seed 0"

run() {
    local nom="$1"; shift
    echo "=== $nom  ($(date +%H:%M)) ===" | tee -a "$LOG"
    # shellcheck disable=SC2086
    python tools/train.py --data "$DATA" $COMMON "$@" \
        --out "engine/bin/B_${nom}.nnue" 2>&1 \
        | grep -E "positions lues|RESULTAT|reseau ecrit" | tee -a "$LOG"
    echo | tee -a "$LOG"
}

echo "### balayage lance le $(date '+%Y-%m-%d %H:%M')" | tee -a "$LOG"

# 1. Reference : l'architecture installee, au nouveau volume. Sans elle, aucune
#    des autres mesures ne veut dire quoi que ce soit.
run plain      --hidden 512  --buckets plain --mirror 0 --out-buckets 4 --activation crelu

# 2. La seule variable qui change est l'activation.
run screlu     --hidden 512  --buckets plain --mirror 0 --out-buckets 4 --activation screlu

# 3. Buckets de roi : perdaient encore a 7,5 M, mais l'ecart se resserrait a
#    chaque palier. C'est la variante qui a le plus a gagner du volume.
run k8         --hidden 512  --buckets k8    --mirror 1 --out-buckets 4 --activation crelu --factorizer 1

# 4. Buckets de sortie : 4 etait l'optimum a 1,5 M seulement.
run ob8        --hidden 512  --buckets plain --mirror 0 --out-buckets 8 --activation crelu

# 5. Capacite. Attention : le moteur est compile pour HIDDEN_SIZE=512 et
#    REFUSERA ce fichier. La mesure d'erreur reste valable ; une recompilation
#    ne se justifie que si elle gagne nettement.
run hl1024     --hidden 1024 --buckets plain --mirror 0 --out-buckets 4 --activation crelu

echo "### balayage termine le $(date '+%Y-%m-%d %H:%M')" | tee -a "$LOG"
grep -E "^===|RESULTAT" "$LOG" | tail -40
