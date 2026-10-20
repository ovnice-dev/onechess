#!/usr/bin/env bash
# SPRT d'Onechess sur un runner GitHub Actions (Linux).
# Le moteur est compile ici ; sa signature bench doit etre celle de Windows
# (ci/signature.txt), sinon on s'arrete : un moteur qui ne cherche pas pareil
# ne teste rien.
#
# Variables : NOM, OPT1, OPT2 (options UCI separees par des espaces),
#             MODE ("nodes" ou "tc:10+0.1"), MINUTES (duree maximale).
set -u
cd "$(dirname "$0")/.."
OUTDIR=ci_out
mkdir -p "$OUTDIR"
NOM=${NOM:-essai}; MODE=${MODE:-nodes}; MINUTES=${MINUTES:-330}

PEXT=""; grep -q ' bmi2 ' /proc/cpuinfo && PEXT="-DUSE_PEXT"
nproc; grep -m1 "model name" /proc/cpuinfo
g++ -std=c++20 -O3 -DNDEBUG -flto=auto -fno-exceptions -march=native $PEXT -pthread \
    -Iengine/fathom engine/src/*.cpp engine/fathom/tbprobe.c -o onechess || { echo "ECHEC compilation"; exit 1; }

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

# Reprise : si un etat precedent a ete depose dans ci_in/, on repart de la.
[ -f "ci_in/$NOM.state" ] && cp "ci_in/$NOM.state" "$OUTDIR/$NOM.state" && echo "reprise de $NOM.state"

timeout -s INT "$(( MINUTES * 60 ))" python3 -u tools/arena.py --e1 ./onechess --e2 ./onechess \
    --name1 "$NOM" --name2 Reference \
    --opt1 EvalFile=reseau/onechess.nnue --opt2 EvalFile=reseau/onechess.nnue $o1 $o2 \
    $CAD --games 20000 --concurrency "$NC" \
    --openings data/uho_085.epd --sprt 0 8 --state "$OUTDIR/$NOM.state" > "$OUTDIR/$NOM.log" 2>&1
echo "code de sortie de l'arene : $?"
grep -E "parties en|^paires|^Elo|verdict" "$OUTDIR/$NOM.log" | tail -4
tail -1 "$OUTDIR/$NOM.log"
