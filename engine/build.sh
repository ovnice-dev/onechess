#!/usr/bin/env bash
# Build Onechess. Usage : ./build.sh [release|debug|profile]
set -e
cd "$(dirname "$0")"

MODE="${1:-release}"
SRC="$(ls src/*.cpp) fathom/tbprobe.c"
# OUT=bin/onechess_x.exe ./build.sh release : binaire de test, sans ecraser
# le binaire PGO installe.
OUT="${OUT:-bin/onechess.exe}"

# StateInfo transporte desormais l'accumulateur NNUE (2 Ko) et la recherche
# descend jusqu'a 246 plis : la pile par defaut de 1 Mo ne suffit plus.
COMMON="-std=c++20 -Wall -Wextra -Wcast-qual -Wshadow -pthread -static -Wl,--stack,67108864"
ARCH="-march=native -mtune=native -DUSE_PEXT"
# Fathom inclut <tbconfig.h> entre chevrons, et compile en C++ :
# son en-tete porte deja son propre extern "C".
INC="-Ifathom"

case "$MODE" in
  release)
    FLAGS="-O3 -DNDEBUG -flto=auto -fno-exceptions"
    ;;
  debug)
    FLAGS="-O1 -g"
    OUT="bin/onechess-debug.exe"
    ;;
  profile)
    FLAGS="-O3 -DNDEBUG -g -fno-omit-frame-pointer"
    OUT="bin/onechess-profile.exe"
    ;;
  pgo)
    # Optimisation guidee par profil : on compile une version instrumentee, on
    # la fait tourner sur le banc, puis on recompile en exploitant le profil.
    # Le compilateur sait alors quelles branches sont prises, ou placer le code
    # chaud, et quoi derouler -- ce qu'aucune heuristique statique ne devine.
    PDIR="$(pwd)/bin/pgo"
    rm -rf "$PDIR"; mkdir -p "$PDIR" bin

    echo "  1/3  compilation instrumentee"
    g++ $COMMON $ARCH $INC ${EXTRA:-} -O3 -DNDEBUG -fprofile-generate="$PDIR"         $SRC -o bin/onechess-gen.exe

    echo "  2/3  execution du profil"
    # Le profil doit refleter l'usage REEL : le moteur joue avec son reseau.
    # Un profil mixe PSQT/NNUE brouille les branches et fait perdre du NPS.
    # NET=... pour profiler avec un reseau d'une autre taille (voir
    # ONE_HIDDEN_SIZE) : un reseau refuse au chargement donnerait un profil
    # PSQT, sans rapport avec l'usage reel.
    PNET="${NET:-$(pwd)/bin/onechess.nnue}"
    printf 'setoption name EvalFile value %s
bench 12
bench 9
' "$PNET"       | ./bin/onechess-gen.exe > "$PDIR/profil.log" 2>&1
    grep -qi "NNUE" "$PDIR/profil.log" || { echo "ECHEC : le reseau $PNET n'a pas ete charge pendant le profil" >&2; exit 1; }

    echo "  3/3  recompilation guidee"
    g++ $COMMON $ARCH $INC ${EXTRA:-} -O3 -DNDEBUG -flto=auto -fno-exceptions         -fprofile-use="$PDIR" -fprofile-correction         $SRC -o "$OUT"
    rm -f bin/onechess-gen.exe
    echo "OK  $OUT  $(ls -lh "$OUT" | awk '{print $5}')"
    exit 0
    ;;
  *)
    echo "mode inconnu: $MODE" >&2; exit 1 ;;
esac

mkdir -p bin
echo "g++ $MODE -> $OUT"
g++ $COMMON $ARCH $INC $FLAGS ${EXTRA:-} $SRC -o "$OUT"
echo "OK  $(ls -lh "$OUT" | awk '{print $5}')"
