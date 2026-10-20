#!/usr/bin/env bash
# Mesure de vitesse honnete entre deux moteurs : alternance ABBA et maximum
# de chaque serie. Une mesure de vitesse sous charge, ou un ordre de passage
# fixe, peuvent dire l'inverse de la verite -- on l'a paye une fois avec la
# fusion de l'accumulateur.
#
#   bash tools/nps_abba.sh A_exe A_net B_exe B_net [tours]
set -u
cd /c/Onechess
AE=$1; AN=$2; BE=$3; BN=$4; T=${5:-4}

nps() {  # exe net -> NPS de bench 12
    printf 'setoption name EvalFile value %s\nbench 12\nquit\n' "$2" \
      | "$1" 2>/dev/null | grep -i "^NPS" | awk '{print $3}'
}

amax=0; bmax=0
for i in $(seq 1 "$T"); do
    a=$(nps "$AE" "$AN"); b=$(nps "$BE" "$BN")   # A puis B
    b2=$(nps "$BE" "$BN"); a2=$(nps "$AE" "$AN") # B puis A : l'ordre s'annule
    for v in $a $a2; do [ "${v:-0}" -gt "$amax" ] && amax=$v; done
    for v in $b $b2; do [ "${v:-0}" -gt "$bmax" ] && bmax=$v; done
    echo "  tour $i : A $a / $a2    B $b / $b2"
done
echo
echo "A ($(basename "$AE")) : $amax NPS"
echo "B ($(basename "$BE")) : $bmax NPS"
python -c "
a=$amax; b=$bmax
print(f'ecart : {100.0*(b-a)/a:+.1f} % pour B')"
