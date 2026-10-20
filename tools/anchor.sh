#!/usr/bin/env bash
# Place le moteur sur l'echelle Elo en le confrontant a Stockfish bride.
# Usage : anchor.sh <elo_stockfish> [parties] [cadence] [moteur]
set -e
SF="$HOME/AppData/Roaming/org.encroissant.app/engines/stockfish/stockfish-windows-x86-64-avx2.exe"
ELO="${1:-2000}"
GAMES="${2:-200}"
TC="${3:-5+0.05}"
ENG="${4:-/c/Onechess/engine/bin/onechess.exe}"

python /c/Onechess/tools/arena.py \
  --e1 "$ENG" --e2 "$SF" \
  --name1 "Onechess" --name2 "SF-$ELO" \
  --opt2 "UCI_LimitStrength=true" --opt2 "UCI_Elo=$ELO" \
  --tc "$TC" --games "$GAMES" --concurrency 6 --hash 16 \
  --openings /c/Onechess/data/openings.epd --quiet
