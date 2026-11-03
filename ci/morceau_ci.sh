#!/usr/bin/env bash
# Un morceau d'un SPRT en lot (voir .github/workflows/sprt_lot.yml) : compile,
# verifie la signature, puis joue PARTIES parties sans arret anticipe.
# Graine des ouvertures : VAGUE*1000 + MORCEAU (jamais deux fois la meme).
set -u
cd "$(dirname "$0")/.."
OUTDIR=ci_out; mkdir -p "$OUTDIR"
NOM=${NOM:-essai}; MODE=${MODE:-nodes}; PARTIES=${PARTIES:-2000}
VAGUE=${VAGUE:-1}; MORCEAU=${MORCEAU:-1}
GRAINE=$(( VAGUE * 1000 + MORCEAU ))

# MODE peut porter des options de compilation du candidat : "tc:10+0.1;x1=-DONE_HIDDEN_SIZE=1024"
EXTRA1=""
EXTRA2=""
# ";x2=..." : options de compilation de la reference (apres x1 s'il y en a un)
if [[ "$MODE" == *";x2="* ]]; then EXTRA2=${MODE#*;x2=}; MODE=${MODE%;x2=*}; fi
if [[ "$MODE" == *";x1="* ]]; then EXTRA1=${MODE#*;x1=}; MODE=${MODE%%;*}; fi
PEXT=""; grep -q ' bmi2 ' /proc/cpuinfo && PEXT="-DUSE_PEXT"
nproc; grep -m1 "model name" /proc/cpuinfo
g++ -std=c++20 -O3 -DNDEBUG -flto=auto -fno-exceptions -march=native $PEXT -pthread \
    -Iengine/fathom engine/src/*.cpp engine/fathom/tbprobe.c -o onechess || { echo "ECHEC compilation"; exit 1; }

E1=./onechess
E2=./onechess
if [ -n "$EXTRA2" ]; then
    g++ -std=c++20 -O3 -DNDEBUG -flto=auto -fno-exceptions -march=native $PEXT -pthread $EXTRA2         -Iengine/fathom engine/src/*.cpp engine/fathom/tbprobe.c -o onechess2 || { echo "ECHEC compilation reference"; exit 1; }
    E2=./onechess2; echo "reference compilee avec : $EXTRA2"
fi
if [ -n "$EXTRA1" ]; then
    g++ -std=c++20 -O3 -DNDEBUG -flto=auto -fno-exceptions -march=native $PEXT -pthread $EXTRA1         -Iengine/fathom engine/src/*.cpp engine/fathom/tbprobe.c -o onechess1 || { echo "ECHEC compilation candidat"; exit 1; }
    E1=./onechess1; echo "candidat compile avec : $EXTRA1"
fi

ulimit -s 65536
n=$(printf 'setoption name EvalFile value reseau/onechess.nnue\nbench 12\nquit\n' | ./onechess | grep 'Noeuds totaux' | awk '{print $4}')
attendu=$(cat ci/signature.txt)
echo "signature Linux $n, attendue $attendu"
[ "$n" = "$attendu" ] || { echo "SIGNATURE DIFFERENTE : arret"; exit 1; }

o1=""; for o in ${OPT1:-}; do o1="$o1 --opt1 $o"; done
o2=""; for o in ${OPT2:-}; do o2="$o2 --opt2 $o"; done
NC=$(nproc)
if [[ "$MODE" == tc:* ]]; then
    CAD="--tc ${MODE#tc:} --hash 64"; NC=$(( NC > 1 ? NC / 2 : 1 ))
else
    CAD="--nodes1 100000 --nodes2 100000 --hash 32"
fi

# arret propre avant la limite de 6 h : l'etat garde les paires jouees
timeout -s INT 20400 python3 -u tools/arena.py --e1 "$E1" --e2 "$E2" \
    --name1 "$NOM" --name2 Reference \
    --opt1 EvalFile=reseau/onechess.nnue --opt2 EvalFile=reseau/onechess.nnue $o1 $o2 \
    $CAD --games "$PARTIES" --concurrency "$NC" --seed "$GRAINE" --quiet \
    --openings data/uho_085.epd --state "$OUTDIR/$NOM-v$VAGUE-m$MORCEAU.state" > "$OUTDIR/$NOM-v$VAGUE-m$MORCEAU.log" 2>&1
echo "code de sortie de l'arene : $?"
tail -6 "$OUTDIR/$NOM-v$VAGUE-m$MORCEAU.log"
