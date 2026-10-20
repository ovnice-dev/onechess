#!/usr/bin/env bash
# File d'attente des SPRT en lot : lance les tests de ci/file_lot.txt les uns
# apres les autres, MAXP a la fois (chaque test occupe MORCEAUX machines).
#   bash ci/file_lot.sh [MAXP=3]
# Une ligne de la file : Nom|options du candidat|bornes|morceaux|parties|vagues max|options reference
# (lignes vides et commentaires # ignores). Un test dont le journal contient
# FIN est fini ; relancer la file reprend exactement ou elle en etait (les
# pilotes reprennent les vagues deja lancees).
cd /c/Onechess
MAXP=${1:-3}
F=ci/file_lot.txt
declare -A actif
while true; do
  # pilotes termines
  for n in "${!actif[@]}"; do kill -0 "${actif[$n]}" 2>/dev/null || unset "actif[$n]"; done
  reste=0
  while IFS='|' read -r nom opts bornes m p vmax opt2; do
    [ -z "$nom" ] && continue; case "$nom" in \#*) continue;; esac
    grep -q " FIN " "ci_resultats/lot/$nom.log" 2>/dev/null && continue
    reste=1
    [ -n "${actif[$nom]}" ] && continue
    [ "${#actif[@]}" -ge "$MAXP" ] && continue
    bash ci/sprt_lot.sh "$nom" "$opts" "${m:-6}" "${p:-1000}" "${bornes:-0 3}" "${vmax:-10}" nodes "$opt2" &
    actif[$nom]=$!
    echo "$(date '+%H:%M') lance $nom"
    sleep 30
  done < "$F"
  [ $reste = 0 ] && [ "${#actif[@]}" = 0 ] && { echo "$(date '+%H:%M') file vide"; exit 0; }
  sleep 120
done
