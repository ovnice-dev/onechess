#!/usr/bin/env bash
# Publie l'etat de main sur le depot PUBLIC (ovnice-dev/onechess), sans les
# documents de travail, qui restent dans le depot prive (ovnice-dev/onechess-prive).
#   bash ci/publier.sh "message"
# Rien n'est reecrit : chaque publication ajoute un enregistrement a la branche
# locale `public`, dont l'arbre est celui de main moins les fichiers PRIVES.
set -e
cd /c/Onechess
PRIVES="ROADMAP.md RECHERCHES.md JOURNAL.md REPRENDRE.md .github/workflows/kaggle_nuit.yml ci/kaggle_nuit/run.py ci/kaggle_nuit/kernel-metadata.json"
MSG=${1:-"Mise a jour"}
export GIT_AUTHOR_NAME=ovnice GIT_COMMITTER_NAME=ovnice
export GIT_AUTHOR_EMAIL=268732922+ovnice-dev@users.noreply.github.com
export GIT_COMMITTER_EMAIL=$GIT_AUTHOR_EMAIL
IDX=$(mktemp); rm -f "$IDX"
GIT_INDEX_FILE=$IDX git read-tree main
GIT_INDEX_FILE=$IDX git rm --cached -q --ignore-unmatch $PRIVES
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
