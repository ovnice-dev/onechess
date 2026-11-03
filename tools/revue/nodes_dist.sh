#!/usr/bin/env bash
# Depassement moyen de "go nodes N" sur les 20 positions du banc.
B="$1"; N="$2"
grep -o '"[^"]*/[^"]*"' ${ONECHESS:-.}/engine/src/uci.cpp | sed -n '2,21p' | tr -d '"' | while read -r fen; do
  printf "setoption name EvalFile value ${ONECHESS:-.}/reseau/onechess.nnue\nucinewgame\nposition fen $fen\ngo nodes $N\nisready\n" | "$B" 2>&1 | grep "info depth" | tail -1 | awk '{for(i=1;i<=NF;i++) if($i=="nodes") print $(i+1)}'
done | awk -v n="$N" '{s+=$1; if($1>m)m=$1; c++} END {printf "N=%d : moyenne %.0f (%+.1f %%), max %d (%+.1f %%) sur %d positions\n", n, s/c, (s/c-n)*100/n, m, (m-n)*100/n, c}'
