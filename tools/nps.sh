#!/usr/bin/env bash
# NPS du banc, meilleur de N passes. Le bruit de mesure ne fait que ralentir :
# le maximum est donc l'estimateur le plus stable de la vitesse reelle.
BIN="${1:-/c/Onechess/engine/bin/onechess.exe}"
NET="${2:-C:/Onechess/engine/bin/onechess.nnue}"
DEPTH="${3:-13}"
N="${4:-5}"
best=0; nodes=0
for i in $(seq "$N"); do
  out=$(printf "setoption name EvalFile value $NET\nbench $DEPTH\n" | "$BIN" 2>&1)
  v=$(echo "$out" | awk '/^NPS/{print $3}')
  nodes=$(echo "$out" | awk '/^Noeuds/{print $4}')
  [ "$v" -gt "$best" ] && best=$v
done
echo "$nodes noeuds   $best nps (meilleur de $N)"
