#!/usr/bin/env bash
# Publie l'etat de main sur le depot PUBLIC (ovnice-dev/onechess), sans les
# documents de travail, qui restent dans le depot prive (ovnice-dev/onechess-prive).
#   bash ci/publier.sh "message"
# Rien n'est reecrit : chaque publication ajoute un enregistrement a la branche
# locale `public`, dont l'arbre est celui de main moins les fichiers PRIVES.
set -e
cd /c/Onechess
PRIVES="sortie ROADMAP.md RECHERCHES.md JOURNAL.md REPRENDRE.md PHASE10.md .github/workflows/kaggle_nuit.yml ci/kaggle_nuit/run.py ci/kaggle_nuit/kernel-metadata.json"
MSG=${1:-"Mise a jour"}
export GIT_AUTHOR_NAME=ovnice GIT_COMMITTER_NAME=ovnice
export GIT_AUTHOR_EMAIL=268732922+ovnice-dev@users.noreply.github.com
export GIT_COMMITTER_EMAIL=$GIT_AUTHOR_EMAIL
IDX=$(mktemp); rm -f "$IDX"
GIT_INDEX_FILE=$IDX git read-tree main
GIT_INDEX_FILE=$IDX git rm -r --cached -q --ignore-unmatch $PRIVES
arbre=$(GIT_INDEX_FILE=$IDX git write-tree); rm -f "$IDX"
if git rev-parse -q --verify refs/heads/public >/dev/null; then
  [ "$(git rev-parse public^{tree})" = "$arbre" ] && { echo "rien de neuf a publier"; exit 0; }
  c=$(echo "$MSG" | git commit-tree "$arbre" -p public)
else
  c=$(echo "$MSG" | git commit-tree "$arbre")
fi
git update-ref refs/heads/public "$c"
for f in $PRIVES; do git ls-tree -r --name-only public | grep -qx "$f" && { echo "ERREUR : $f dans public"; exit 1; }; done
git push public public:main
echo "publie : $(git rev-parse --short public)"

# Depot des outils (ovnice-dev/onechess-tools, 13-11) : la meme branche main, reduite
# aux dossiers d'outils, avec son propre README (sortie/outils/README.md).
OUTILS="ci tools colab lichess data reseau/essais reseau/archives LICENSE .gitattributes"
git remote get-url outils >/dev/null 2>&1 || { echo "(pas de remote outils : depot des outils non publie)"; exit 0; }
IDX=$(mktemp); rm -f "$IDX"
GIT_INDEX_FILE=$IDX git read-tree main
GIT_INDEX_FILE=$IDX git rm -r --cached -q --ignore-unmatch $PRIVES
GIT_INDEX_FILE=$IDX git ls-files | while read -r f; do
  ok=0; for d in $OUTILS; do case "$f" in "$d"|"$d"/*) ok=1;; esac; done
  [ $ok = 1 ] || echo "$f"
done | GIT_INDEX_FILE=$IDX git update-index --force-remove --stdin
readme=$(git hash-object -w sortie/outils/README.md)
GIT_INDEX_FILE=$IDX git update-index --add --cacheinfo 100644,"$readme",README.md
arbre=$(GIT_INDEX_FILE=$IDX git write-tree); rm -f "$IDX"
if git rev-parse -q --verify refs/heads/outils >/dev/null; then
  [ "$(git rev-parse outils^{tree})" = "$arbre" ] && { echo "outils : rien de neuf"; exit 0; }
  c=$(echo "$MSG" | git commit-tree "$arbre" -p outils)
else
  c=$(echo "$MSG" | git commit-tree "$arbre")
fi
git update-ref refs/heads/outils "$c"
for f in $PRIVES engine; do git ls-tree -r --name-only outils | grep -q "^$f" && { echo "ERREUR : $f dans outils"; exit 1; }; done
git push outils outils:main
echo "outils publies : $(git rev-parse --short outils)"
