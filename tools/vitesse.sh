#!/usr/bin/env bash
# Campagne de mesure de la vitesse. EXIGE LA MACHINE SEULE.
#
# Deux lectures de NPS du meme binaire ont donne 1 346 981 et 1 425 446, soit
# 6 % d'ecart. Une mesure unique ne vaut donc rien : chaque point est repete
# cinq fois et on garde la MEDIANE, insensible a une valeur aberrante comme
# le serait la moyenne.
#
# Le nombre de noeuds a profondeur fixe, lui, est deterministe : il sert de
# preuve qu'un changement ne touche QUE la vitesse et pas la recherche.
set -u
cd /c/Onechess

REPEATS=${REPEATS:-5}
DEPTH=${DEPTH:-12}
NET=C:/Onechess/engine/bin/onechess.nnue

libre() {
    local n
    n=$(powershell.exe -NoProfile -Command "@(Get-Process onechess,stockfish-windows-x86-64-avx2 -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r')
    [ "${n:-0}" -eq 0 ]
}

# Rend "mediane_nps noeuds" pour un binaire et un nombre de fils donnes.
mesure() {
    local exe="$1" threads="${2:-1}"
    local nps=() noeuds=""
    for _ in $(seq 1 "$REPEATS"); do
        local out
        out=$(printf 'setoption name Threads value %s\nsetoption name EvalFile value %s\nisready\nbench %s\nquit\n' \
                     "$threads" "$NET" "$DEPTH" | "$exe" 2>&1)
        nps+=("$(echo "$out" | awk '/^NPS/ {print $3}')")
        noeuds=$(echo "$out" | awk '/^Noeuds totaux/ {print $4}')
    done
    printf '%s\n' "${nps[@]}" | sort -n | awk -v n="$noeuds" '{a[NR]=$1} END {printf "%d %s\n", a[int((NR+1)/2)], n}'
}

if ! libre; then
    echo "REFUS : des moteurs tournent deja. Une mesure de vitesse sous charge ne vaut rien." >&2
    exit 1
fi

echo "=== campagne vitesse  $(date '+%Y-%m-%d %H:%M')  ($REPEATS repetitions, profondeur $DEPTH) ==="
echo

echo "--- 1. compilation guidee par profil (PGO) ---"
cp engine/bin/onechess.exe engine/bin/onechess_sans_pgo.exe
read -r nps_a noeuds_a <<< "$(mesure engine/bin/onechess_sans_pgo.exe 1)"
printf "  sans PGO : %'d NPS   (%s noeuds)\n" "$nps_a" "$noeuds_a"

bash engine/build.sh pgo > /dev/null 2>&1 || { echo "  ECHEC de la compilation PGO" >&2; exit 1; }
read -r nps_b noeuds_b <<< "$(mesure engine/bin/onechess.exe 1)"
printf "  avec PGO : %'d NPS   (%s noeuds)\n" "$nps_b" "$noeuds_b"
awk -v a="$nps_a" -v b="$nps_b" 'BEGIN {printf "  gain : %+.1f %%\n", 100*(b-a)/a}'

if [ "$noeuds_a" != "$noeuds_b" ]; then
    echo "  ATTENTION : le nombre de noeuds a change ($noeuds_a -> $noeuds_b)."
    echo "  Le PGO ne doit toucher QUE la vitesse. A elucider avant d'adopter."
else
    echo "  noeuds identiques : le PGO ne change que la vitesse, comme attendu"
fi

echo
echo "--- 2. passage a l'echelle en fils (6 coeurs physiques, 12 logiques) ---"
printf "  %-8s %14s %10s %12s\n" "fils" "NPS median" "gain" "efficacite"
base=""
for t in 1 2 4 6 8 12; do
    read -r nps _ <<< "$(mesure engine/bin/onechess.exe "$t")"
    [ -z "$base" ] && base="$nps"
    awk -v t="$t" -v n="$nps" -v b="$base" \
        'BEGIN {printf "  %-8s %14d %9.2fx %11.0f %%\n", t, n, n/b, 100*(n/b)/t}'
done
echo
echo "L'efficacite est le gain rapporte au nombre de fils. Elle chute quand la"
echo "machine devient limitee par la memoire : un Thread pese 9,61 Mo pour"
echo "12 Mo de cache L3 PARTAGE, donc six fils demandent 58 Mo."
echo
echo "Rappel : le NPS n'est pas la force. Un gain de vitesse doit encore etre"
echo "converti en Elo par un match, a cadence reelle."
