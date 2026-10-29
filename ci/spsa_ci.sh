#!/usr/bin/env bash
# Un SPSA sur une machine GitHub (voir .github/workflows/spsa_lot.yml) :
# compile, verifie la signature, puis joue ITERS paires a 100 000 noeuds,
# ou au temps si CADENCE est donnee (ex. 4+0.04) : un reglage a noeuds fixes
# ne survit pas toujours au temps (SPSA6 : H1 a noeuds, H0 au temps).
# Arret de securite avant la limite de 6 h ; spsa.py ecrit son resultat toutes
# les 100 paires, rien n'est perdu.
set -u
cd "$(dirname "$0")/.."
mkdir -p ci_out
PEXT=""; grep -q ' bmi2 ' /proc/cpuinfo && PEXT="-DUSE_PEXT"
nproc; grep -m1 "model name" /proc/cpuinfo
g++ -std=c++20 -O3 -DNDEBUG -flto=auto -fno-exceptions -march=native $PEXT -pthread \
    -Iengine/fathom engine/src/*.cpp engine/fathom/tbprobe.c -o onechess || { echo "ECHEC compilation"; exit 1; }
ulimit -s 65536
n=$(printf 'setoption name EvalFile value reseau/onechess.nnue\nbench 12\nquit\n' | ./onechess | grep 'Noeuds totaux' | awk '{print $4}')
attendu=$(cat ci/signature.txt)
echo "signature Linux $n, attendue $attendu"
[ "$n" = "$attendu" ] || { echo "SIGNATURE DIFFERENTE : arret"; exit 1; }
S=$(( ${GRAINE:-100} + ${MORCEAU:-1} ))
if [ -n "${CADENCE:-}" ]; then MODE="--tc $CADENCE"; else MODE="--nodes 100000"; fi
echo "mode : $MODE"
timeout 340m python -u tools/spsa.py --engine ./onechess --net reseau/onechess.nnue \
    --openings data/uho_085.epd --out ci_out/spsa_m${MORCEAU}.json --iters "${ITERS:-1900}" \
    --seed $S --workers "$(nproc)" $MODE --hash 16 --params $(cat "${PARAMS}") \
    2>&1 | tee ci_out/spsa_m${MORCEAU}.log | grep -E "paires|SPSA|Erreur|Error" | tail -40
ls -la ci_out
