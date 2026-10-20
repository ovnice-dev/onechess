#!/usr/bin/env bash
# Rend la main des qu'un SPRT GitHub Actions se termine (succes ou echec),
# avec son verdict.
#   bash ci/veille_gh.sh
G="/c/Program Files/GitHub CLI/gh.exe"
cd /c/Onechess
liste() { "$G" run list --limit 30 --json databaseId,status -q '.[] | select(.status!="completed") | .databaseId' 2>/dev/null | sort; }
avant=$(liste)
[ -z "$avant" ] && { echo "aucun run en cours"; exit 0; }
while true; do
  sleep 300
  apres=$(liste)
  finis=$(comm -23 <(echo "$avant") <(echo "$apres"))
  if [ -n "$finis" ]; then
    for id in $finis; do
      d=/c/Onechess/ci_resultats/$id; mkdir -p "$d"
      "$G" run download "$id" -D "$d" >/dev/null 2>&1
      echo "=== run $id termine ($("$G" run view "$id" --json conclusion -q .conclusion)) ==="
      grep -hE "parties en|^Elo|verdict|SIGNATURE|ECHEC" "$d"/*/*.log 2>/dev/null | tail -3
    done
    exit 0
  fi
  avant=$apres
done
