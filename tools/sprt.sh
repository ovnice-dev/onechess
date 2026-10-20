#!/usr/bin/env bash
# Teste un changement de constante par SPRT, contre le moteur de reference.
#
#   ./sprt.sh RfpMargin 95                     teste RfpMargin=95 contre le defaut
#   ./sprt.sh LmrDivisor 210 -3 3              avec des bornes SPRT explicites
#   ./sprt.sh RfpMargin 95 0 4 40000 n25000    a 25 000 noeuds fixes par coup
#
# Le moteur est le meme binaire des deux cotes : seule l'option change. Cela
# elimine toute difference de compilation, et rend le resultat attribuable a
# la constante et a elle seule.
#
# La forme "nN" remplace l'horloge par un nombre de noeuds fixe. Les deux
# moteurs cherchent alors exactement le meme nombre de noeuds quelle que soit
# la charge de la machine : le resultat ne depend que de la qualite des coups,
# et le test peut tourner pendant un entrainement sans etre fausse. En
# contrepartie il ne mesure plus rien de ce qui touche a la vitesse ni a la
# gestion du temps.
set -e

NAME="$1"
VALUE="$2"
ELO0="${3:-0}"
ELO1="${4:-4}"
GAMES="${5:-40000}"
LIMIT="${6:-8+0.08}"
CONC="${7:-10}"

ENG=/c/Onechess/engine/bin/onechess.exe
NET=C:/Onechess/engine/bin/onechess.nnue

if [ -z "$NAME" ] || [ -z "$VALUE" ]; then
  echo "usage: sprt.sh <NomOption> <valeur> [elo0] [elo1] [parties] [cadence|nNOEUDS] [concurrence]" >&2
  exit 1
fi

if [[ "$LIMIT" =~ ^n([0-9]+)$ ]]; then
    N="${BASH_REMATCH[1]}"
    MODE=(--nodes1 "$N" --nodes2 "$N")
    echo "SPRT  $NAME = $VALUE   bornes [$ELO0, $ELO1]   $N noeuds fixes"
else
    MODE=(--tc "$LIMIT")
    echo "SPRT  $NAME = $VALUE   bornes [$ELO0, $ELO1]   cadence $LIMIT"
fi

python /c/Onechess/tools/arena.py \
  --e1 "$ENG" --e2 "$ENG" \
  --name1 "$NAME=$VALUE" --name2 "defaut" \
  --opt1 "EvalFile=$NET" --opt1 "$NAME=$VALUE" \
  --opt2 "EvalFile=$NET" \
  "${MODE[@]}" --games "$GAMES" --concurrency "$CONC" --hash 32 \
  --openings /c/Onechess/data/openings.epd \
  --sprt "$ELO0" "$ELO1" --quiet
