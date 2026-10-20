#!/usr/bin/env bash
# Pilote un SPRT en lot sur GitHub Actions, vague apres vague, jusqu'au verdict.
#   bash ci/sprt_lot.sh Nom "Opt1=1 Opt2=3" [morceaux=8] [parties=2000] [bornes="0 5"] [vagues max=8] [mode=nodes] ["opt reference"]
# Journal : ci_resultats/lot/Nom.log (lu par tools/panneau.sh).
G="/c/Program Files/GitHub CLI/gh.exe"
REPO=${REPO:-ovnice-dev/onechess}
cd /c/Onechess
NOM=$1; OPT1=$2; M=${3:-8}; P=${4:-2000}; BORNES=${5:-"0 5"}; VMAX=${6:-8}; MODE=${7:-nodes}; OPT2=${8:-}
D=ci_resultats/lot/$NOM; mkdir -p "$D"; J=ci_resultats/lot/$NOM.log
v=1; while [ -d "$D/v$v" ]; do v=$((v+1)); done      # reprise : vagues deja rapatriees
echo "$(date '+%d-%m %H:%M') DEBUT $NOM  opt1=[$OPT1] opt2=[$OPT2] $M x $P parties, bornes [$BORNES], mode $MODE" >> "$J"
while [ $v -le "$VMAX" ]; do
  "$G" workflow run sprt_lot.yml -R "$REPO" -f nom="$NOM" -f opt1="$OPT1" -f opt2="$OPT2" -f mode="$MODE" \
       -f morceaux="$M" -f parties="$P" -f vague="$v" -f bornes="$BORNES" || { echo "ECHEC lancement" >> "$J"; exit 1; }
  sleep 20; id=""
  for i in 1 2 3 4 5 6; do
    id=$("$G" run list -R "$REPO" --workflow sprt_lot.yml --limit 20 --json databaseId,displayTitle \
         -q ".[] | select(.displayTitle | startswith(\"$NOM (vague $v,\")) | .databaseId" | head -1)
    [ -n "$id" ] && break; sleep 15
  done
  [ -z "$id" ] && { echo "run introuvable" >> "$J"; exit 1; }
  echo "$(date '+%H:%M') vague $v lancee : run $id" >> "$J"
  while [ "$("$G" run view "$id" -R "$REPO" --json status -q .status)" != completed ]; do sleep 120; done
  "$G" run download "$id" -R "$REPO" -p 'etat-*' -D "$D/v$v" >/dev/null 2>&1
  b=$(python ci/bilan_lot.py --bornes $BORNES "$D")
  echo "$(date '+%H:%M') apres vague $v :" >> "$J"; echo "$b" | sed 's/^/    /' >> "$J"
  case "$b" in *"VERDICT H1"*|*"VERDICT H0"*) echo "$(date '+%H:%M') FIN $NOM $(echo "$b" | tail -1)" >> "$J"; exit 0;; esac
  v=$((v+1))
done
echo "$(date '+%H:%M') FIN $NOM VERDICT NON CONCLU apres $VMAX vagues" >> "$J"
