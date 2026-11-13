#!/usr/bin/env bash
# Pilote de l'entrainement continu, sur GitHub (toutes les heures, hebdo.yml),
# sans le PC. Etat porte par les releases du depot :
#   - candidat-K (pre-release, deposee par le relais Kaggle) : onechess.nnue ;
#   - vX.Y (release) : une version d Onechess ; un reseau adopte devient la version suivante.
# Pour chaque candidat sans verdict : si sa vague en cours est finie, bilan ;
# H1 -> release de la version suivante (le bot d'Oracle la prend tout seul) ; H0 ou trop de
# vagues -> verdict ecrit dans la pre-release ; sinon, vague suivante. Le match
# oppose le candidat au dernier reseau adopte, au temps, binaires identiques.
#   GH_TOKEN, REPO dans l'environnement ; HEBDO_X : options de compilation (ex. -DONE_HIDDEN_SIZE=1024)
set -eu
REPO=${REPO:-ovnice-dev/onechess}
BORNES="0 3"; MORCEAUX=10; PARTIES=150; VMAX=20; MODE="tc:10+0.1"
X=${HEBDO_X:-}
[ -n "$X" ] && MODE="$MODE;x1=$X;x2=$X"
url() { gh release view "$1" -R "$REPO" --json assets -q ".assets[] | select(.name==\"$2\") | .url"; }

# Reference : la derniere version publiee (vX.Y, depuis la 1.0 du 13-11) ; un reseau de la
# semaine adopte devient la version suivante (v1.0 -> v1.1)
ref=$(gh release list -R "$REPO" --exclude-pre-releases --limit 50 --json tagName,publishedAt \
      -q '[.[] | select(.tagName | test("^v[0-9]+[.][0-9]+$"))] | sort_by(.publishedAt) | last | .tagName' || true)
[ -n "$ref" ] && [ "$ref" != "null" ] || { echo "aucune version publiee (release vX.Y) : rien a juger"; exit 0; }
maj=${ref#v}; maj=${maj%%.*}; mineur=${ref##*.}
suivante="v$maj.$((mineur + 1))"
REF_URL=$(url "$ref" onechess.nnue)
echo "reference : $ref"

for cand in $(gh release list -R "$REPO" --limit 50 --json tagName,isPrerelease -q '.[] | select(.isPrerelease) | .tagName' | grep '^candidat-' || true); do
  corps=$(gh release view "$cand" -R "$REPO" --json body -q .body)
  case "$corps" in *VERDICT*) continue;; esac
  K=${cand#candidat-}; NOM="Hebdo$K"
  CAND_URL=$(url "$cand" onechess.nnue)
  runs() { gh run list -R "$REPO" --workflow sprt_lot.yml --limit 200 --json databaseId,displayTitle,status,conclusion \
             -q ".[] | select(.displayTitle | startswith(\"$NOM (vague \")) | select(.conclusion != \"cancelled\" and .conclusion != \"failure\" and .conclusion != \"startup_failure\") | \"\(.databaseId) \(.status)\""; }
  if runs | grep -qv ' completed$'; then echo "$NOM : vague en cours"; continue; fi
  D=etats/$NOM; rm -rf "$D"; mkdir -p "$D"
  v=0
  for id in $(runs | awk '{print $1}'); do
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
      # Adoption : signature du moteur avec ce reseau, release de la version suivante
      curl -sSfL "$CAND_URL" -o onechess.nnue
      g++ -std=c++20 -O3 -DNDEBUG -flto=auto -fno-exceptions -march=native -pthread $X -Iengine/fathom \
          engine/src/*.cpp engine/fathom/tbprobe.c -o moteur
      sig=$(printf 'setoption name EvalFile value onechess.nnue\nbench 12\nquit\n' | ./moteur | grep 'Noeuds totaux' | awk '{print $4}')
      echo "$sig" > signature.txt
      gh release create "$suivante" onechess.nnue signature.txt -R "$REPO" --target main \
         --title "Onechess ${suivante#v}" \
         --notes "New network from continuous training (week $K). Time-control match against $ref: $(echo "$b" | tail -2 | head -1). bench 12 signature: $sig. Executables: build from source, or use those of the previous release with this network."
      gh release edit "$cand" -R "$REPO" --notes "$corps
VERDICT H1 : adopte comme $suivante. $(echo "$b" | tail -2 | head -1)"
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
