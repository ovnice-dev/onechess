#!/usr/bin/env bash
# Test A/B de la QUALITE des donnees, a volume egal.
#
# Deux reseaux entraines avec la meme recette -- celle du reseau installe --
# sur les N premieres positions de chaque source, puis un match entre eux avec
# le meme binaire : seul le reseau change. Seul le match tranche : l'erreur en
# cp ne se compare pas entre distributions (lecon du lot Lichess, -464 Elo avec
# des chiffres de validation trompeurs).
#
# Il repond a la question qui conditionne la campagne Elite, et sans attendre
# 50 M : a nombre de positions egal, une position Elite vaut-elle mieux qu'une
# position du lot actuel ?
#
# Entrainement sur la carte graphique (~10 min par reseau a 7,5 M) ; le match
# a noeuds fixes supporte une machine chargee.
#
#   N=7500000 bash tools/ab_donnees.sh              entrainement + match
#   N=5000000 PHASE=train bash tools/ab_donnees.sh  entrainement seul
#   N=5000000 PHASE=match CONC=2 bash tools/ab_donnees.sh
set -eu
cd /c/Onechess

N=${N:-7500000}
PHASE=${PHASE:-all}
CONC=${CONC:-3}
GAMES=${GAMES:-1000}
TAG=$((N / 100000))
TAG="${TAG%?}.${TAG: -1}M"            # 7500000 -> 7.5M
RECETTE="--hidden 512 --buckets plain --mirror 0 --out-buckets 4 --factorizer 0 --epochs 40"
mkdir -p data/_tmp

train() {  # nom motif
    # Un cache par source ET par taille : un cache existant est relu tel quel.
    python -u tools/train.py --data "$2" --limit "$N" --cache "data/_tmp/cache_ab_$1_$TAG.npz" \
        --out "engine/bin/ab_$1_$TAG.nnue" $RECETTE > "data/ab_$1_$TAG.log" 2>&1
    grep RESULTAT "data/ab_$1_$TAG.log"
}

if [ "$PHASE" = all ] || [ "$PHASE" = train ]; then
    lot=$(find data/nnue_elite -name '*.bin' -printf '%s\n' | awk '{s+=$1} END {print int(s/32)}')
    [ "$lot" -ge "$N" ] || { echo "lot Elite : $lot positions, il en faut $N"; exit 1; }
    echo "### A/B donnees $TAG  $(date '+%Y-%m-%d %H:%M')"
    train big   "data/nnue_big/*.bin"
    train elite "data/nnue_elite/*/*.bin"
fi

if [ "$PHASE" = all ] || [ "$PHASE" = match ]; then
    python tools/arena.py --e1 engine/bin/onechess.exe --e2 engine/bin/onechess.exe \
        --name1 "Elite_$TAG" --name2 "Big_$TAG" \
        --opt1 "EvalFile=C:/Onechess/engine/bin/ab_elite_$TAG.nnue" \
        --opt2 "EvalFile=C:/Onechess/engine/bin/ab_big_$TAG.nnue" \
        --nodes1 100000 --nodes2 100000 --games "$GAMES" --concurrency "$CONC" --hash 32 \
        --openings data/openings.epd --state "data/ab_$TAG.state" >> "data/ab_$TAG.log" 2>&1
    tail -n 2 "data/ab_$TAG.log"
fi
