#!/usr/bin/env bash
# Pilote de l'entrainement continu, sur GitHub (toutes les heures, hebdo.yml),
# sans le PC. Etat porte par les releases du depot :
#   - candidat-K (pre-release, deposee par le relais Kaggle) : onechess.nnue ;
#   - reseau-K (release) : un reseau adopte, onechess.nnue + signature.txt.
# Pour chaque candidat sans verdict : si sa vague en cours est finie, bilan ;
# H1 -> release reseau-K (le bot d'Oracle la prend tout seul) ; H0 ou trop de
# vagues -> verdict ecrit dans la pre-release ; sinon, vague suivante. Le match
# oppose le candidat au dernier reseau adopte, au temps, binaires identiques.
#   GH_TOKEN, REPO dans l'environnement ; HEBDO_X : options de compilation (ex. -DONE_HIDDEN_SIZE=1024)
set -eu
REPO=${REPO:-ovnice-dev/onechess}
BORNES="0 3"; MORCEAUX=10; PARTIES=150; VMAX=20; MODE="tc:10+0.1"
X=${HEBDO_X:-}
[ -n "$X" ] && MODE="$MODE;x1=$X;x2=$X"
url() { gh release view "$1" -R "$REPO" --json assets -q ".assets[] | select(.name==\"$2\") | .url"; }

ref=$(gh release list -R "$REPO" --exclude-pre-releases --limit 50 --json tagName -q '.[].tagName' | grep '^reseau-' | head -1 || true)
[ -n "$ref" ] || { echo "aucun reseau adopte (release reseau-*) : rien a juger"; exit 0; }
REF_URL=$(url "$ref" onechess.nnue)
echo "reference : $ref"

for cand in $(gh release list -R "$REPO" --limit 50 --json tagName,isPrerelease -q '.[] | select(.isPrerelease) | .tagName' | grep '^candidat-' || true); do
  corps=$(gh release view "$cand" -R "$REPO" --json body -q .body)
  case "$corps" in *VERDICT*) continue;; esac
  K=${cand#candidat-}; NOM="Hebdo$K"
  CAND_URL=$(url "$cand" onechess.nnue)
  runs() { gh run list -R "$REPO" --workflow sprt_lot.yml --limit 200 --json databaseId,displayTitle,status,conclusion \
             -q ".[] | select(.displayTitle | startswith(\"$NOM (vague \")) | select(.conclusion != \"cancelled\")"; }
  if runs | grep -q '"status":"\(queued\|in_progress\|waiting\)"'; then echo "$NOM : vague en cours"; continue; fi
  D=etats/$NOM; rm -rf "$D"; mkdir -p "$D"
  v=0
  for id in $(runs | python3 -c "import sys,json;[print(json.loads(l)['databaseId']) for l in sys.stdin if l.strip()]"); do
    v=$((v+1)); gh run download "$id" -R "$REPO" -p 'etat-*' -D "$D/r$id" >/dev/null 2>&1 || true
  done
  if [ "$v" -gt 0 ]; then
    b=$(python3 ci/bilan_lot.py --bornes $BORNES "$D")
    echo "$NOM apres $v vague(s) :"; echo "$b"
  else
    b=""
  fi
  case "$b" in
    *"VERDICT H1"*)
      # Adoption : signature du moteur avec ce reseau, release reseau-K
      curl -sSfL "$CAND_URL" -o onechess.nnue
      g++ -std=c++20 -O3 -DNDEBUG -flto=auto -fno-exceptions -march=native -pthread $X -Iengine/fathom \
          engine/src/*.cpp engine/fathom/tbprobe.c -o moteur
      sig=$(printf 'setoption name EvalFile value onechess.nnue\nbench 12\nquit\n' | ./moteur | grep 'Noeuds totaux' | awk '{print $4}')
      echo "$sig" > signature.txt
      gh release create "reseau-$K" onechess.nnue signature.txt -R "$REPO" --target main \
         --title "Onechess, reseau de la semaine $K" \
         --notes "Reseau de l'entrainement continu, semaine $K. Match au temps contre $ref : $(echo "$b" | tail -2 | head -1). Signature bench 12 : $sig."
      gh release edit "$cand" -R "$REPO" --notes "$corps
VERDICT H1 : adopte comme reseau-$K. $(echo "$b" | tail -2 | head -1)"
      ;;
    *"VERDICT H0"*)
      gh release edit "$cand" -R "$REPO" --notes "$corps
VERDICT H0 : non adopte. $(echo "$b" | tail -2 | head -1)"
      ;;
    *)
      if [ "$v" -ge "$VMAX" ]; then
        gh release edit "$cand" -R "$REPO" --notes "$corps
VERDICT NON CONCLU apres $v vagues : non adopte. $(echo "$b" | tail -2 | head -1)"
      else
        gh workflow run sprt_lot.yml -R "$REPO" -f nom="$NOM" -f opt1="EvalFile=$CAND_URL" -f opt2="EvalFile=$REF_URL" \
           -f mode="$MODE" -f morceaux="$MORCEAUX" -f parties="$PARTIES" -f vague="$((v+1))" -f bornes="$BORNES"
        echo "$NOM : vague $((v+1)) lancee"
      fi
      ;;
  esac
done
