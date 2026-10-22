#!/usr/bin/env bash
# Avant d'eteindre le PC : lance d'avance des vagues sur GitHub pour que les
# tests continuent toute la nuit. Les pilotes les reprendront au rallumage
# (ils retrouvent une vague lancee par son nom, depuis ci/ref_depuis.txt).
#   bash ci/nuit_lot.sh [nombre de tests=12] [vagues d'avance=3]
cd /c/Onechess
G="/c/Program Files/GitHub CLI/gh.exe"; REPO=ovnice-dev/onechess
NT=${1:-12}; NV=${2:-3}
DEPUIS=$(cat ci/ref_depuis.txt)
deja=$("$G" run list -R $REPO --workflow sprt_lot.yml --limit 200 --json displayTitle,createdAt,conclusion \
       -q ".[] | select(.createdAt >= \"$DEPUIS\") | select(.conclusion != \"cancelled\") | .displayTitle")
k=0
while IFS='|' read -r nom opts bornes m p vmax opt2 mode; do
  [ -z "$nom" ] && continue; case "$nom" in \#*) continue;; esac
  grep -q " FIN " "ci_resultats/lot/$nom.log" 2>/dev/null && continue
  k=$((k+1)); [ $k -gt "$NT" ] && break
  # premiere vague pas encore lancee
  v=1; while echo "$deja" | grep -q "^$nom (vague $v,"; do v=$((v+1)); done
  for i in $(seq 1 "$NV"); do
    [ $v -gt "${vmax:-10}" ] && break
    "$G" workflow run sprt_lot.yml -R $REPO -f nom="$nom" -f opt1="$opts" -f opt2="$opt2" \
        -f mode="${mode:-nodes}" -f morceaux="${m:-6}" -f parties="${p:-1000}" -f vague="$v" \
        -f bornes="${bornes:-0 3}" >/dev/null && echo "$nom vague $v lancee"
    v=$((v+1)); sleep 3
  done
done < ci/file_lot.txt
