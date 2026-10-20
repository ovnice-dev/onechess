#!/usr/bin/env bash
# Tournoi d'ancrage : Onechess contre des moteurs de note CCRL 40/15 connue.
#
# Le seul ancrage absolu jusqu'ici etait un match contre Stockfish 18 a 100 k
# noeuds (-511,5 +/- 131,6) : un adversaire 400 Elo plus fort donne des
# resultats tres desequilibres, donc une barre d'erreur enorme, et des noeuds
# fixes ne se comparent pas entre moteurs differents. On fait ici comme les
# listes de classement : plusieurs adversaires proches de notre niveau, au
# temps, un coeur chacun. tools/ccrl_rating.py combine les matchs en une note.
#
# Notes CCRL 40/15 mono-coeur, liste du 10 septembre 2026 :
#   Leorik 3.0          3225 +/- 12     Weiss 2.0        3264 +/- 10
#   Avalanche 2.1.0     3350 +/-  9     Reckless 0.7.0   3421 +/- 12
#   RubiChess 20240817  3569 +/-  7
# Binaires officiels des auteurs (GitHub), dans engines/ ; RubiChess est celui
# d'En Croissant.
#
# Exige une machine libre : c'est une mesure au temps. Un match par
# adversaire, une partie a la fois chacun, soit 5 parties simultanees sur
# 6 coeurs. Pas de tables de finales pour personne.
#
# Reprise : chaque match a son fichier d'etat ; relancer le script reprend
# chaque match la ou il s'etait arrete.
#
#   bash tools/gauntlet.sh            (GAMES=300 TC=10+0.1 par defaut)
set -u
cd /c/Onechess

ONE=engine/bin/onechess.exe
NET=C:/Onechess/engine/bin/onechess.nnue
TC=${TC:-10+0.1}
GAMES=${GAMES:-300}
OUT=data/gauntlet
RUBI=$HOME/AppData/Roaming/org.encroissant.app/engines/RubiChess-20240817/windows/RubiChess-20240817_x86-64-avx2.exe

mkdir -p "$OUT"
echo "### tournoi d'ancrage lance $(date '+%Y-%m-%d %H:%M')   cadence $TC   $GAMES parties par adversaire" | tee -a "$OUT/tournoi.log"

run() {  # nom chemin
    python -u tools/arena.py --e1 "$ONE" --e2 "$2" --name1 Onechess --name2 "$1" \
        --opt1 "EvalFile=$NET" --tc "$TC" --games "$GAMES" --concurrency 1 --hash 64 \
        --openings data/openings.epd --state "$OUT/$1.state" >> "$OUT/$1.log" 2>&1 &
}

run Leorik-3.0         engines/leorik/Leorik-3.0.1.exe
run Weiss-2.0          engines/weiss/weiss-pext.exe
run Avalanche-2.1.0    engines/avalanche-2.1.0.exe
run Reckless-0.7.0     engines/reckless-0.7.0.exe
run RubiChess-20240817 "$RUBI"

wait
echo "### tournoi d'ancrage termine $(date '+%Y-%m-%d %H:%M')" | tee -a "$OUT/tournoi.log"
python tools/ccrl_rating.py | tee -a "$OUT/tournoi.log"
