#!/usr/bin/env bash
# Sauvegarde de ce qui ne se rachete pas avec du temps.
#
# La cle ecrit a 15 Mo/s : elle ne peut pas servir d'espace de travail, mais
# elle est parfaite pour de l'archive sequentielle. On n'y met donc QUE ce dont
# la perte couterait des heures de calcul ou serait definitive.
#
# Ce qu'on ne sauvegarde PAS, et pourquoi :
#   cache_full.npz  1,2 Go, reconstruit en 5 min depuis les .bin
#   syzygy/         0,9 Go, retelechargeable
#   Gaviota/        6,5 Go, retelechargeable
#   bin/*.exe       recompilable en 2 min
set -u
SRC=/c/Onechess
DST=/e/Onechess
t0=$(date +%s)

mkdir -p "$DST"/{donnees,reseaux,code,source}

echo "1/4  positions etiquetees (537 Mo, ~15 h de calcul)"
mkdir -p "$DST/donnees/nnue_big" "$DST/donnees/nnue"
cp -u "$SRC"/data/nnue_big/*.bin "$DST/donnees/nnue_big/" 2>/dev/null
cp -u "$SRC"/data/nnue/*.bin     "$DST/donnees/nnue/"     2>/dev/null

echo "2/4  reseaux entraines (25 Mo, ~4 h de GPU)"
cp -u "$SRC"/engine/bin/*.nnue "$DST/reseaux/"

echo "3/4  code, journal, outillage (0,7 Mo)"
mkdir -p "$DST/code/src" "$DST/code/fathom" "$DST/code/tools"
cp -u "$SRC"/engine/src/*    "$DST/code/src/"
cp -u "$SRC"/engine/fathom/* "$DST/code/fathom/"
cp -u "$SRC"/tools/*         "$DST/code/tools/"
cp -u "$SRC"/engine/build.sh "$DST/code/"
cp -u "$SRC"/*.md            "$DST/code/"
cp -u "$SRC"/data/openings.epd "$SRC"/data/spsa_*.json "$DST/code/" 2>/dev/null

# Le CSV d'origine de l'auteur : il etait sur la cle, la cle a ete reformatee,
# il n'en existe plus qu'UNE seule copie sur C:. C'est sa donnee, pas la notre.
echo "4/4  corpus d'origine (1,7 Go) -- seule copie existante"
cp -u "$SRC/data/positions.csv" "$DST/source/" 2>/dev/null

date '+%Y-%m-%d %H:%M' > "$DST/DERNIERE_SAUVEGARDE.txt"
{
  echo "Sauvegarde Onechess"
  echo
  echo "donnees/  positions etiquetees, 16 049 828 au total (~15 h de Stockfish)"
  echo "reseaux/  tous les reseaux entraines, le plus fort est celui installe"
  echo "code/     sources, outils, JOURNAL.md et REPRENDRE.md"
  echo "source/   le CSV Lichess d'origine -- seule copie apres reformatage de la cle"
  echo
  echo "Restaurer : recopier vers C:\\Onechess, puis"
  echo "  bash engine/build.sh release"
  echo "  python tools/check_nnue.py"
} >> "$DST/DERNIERE_SAUVEGARDE.txt"

echo
echo "termine en $(( ($(date +%s) - t0) / 60 )) min $(( ($(date +%s) - t0) % 60 )) s"
du -sh "$DST" 2>/dev/null
