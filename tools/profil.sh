#!/usr/bin/env bash
# Ou passe le temps dans le moteur ? EXIGE LA MACHINE SEULE.
#
# Il n'y a pas de profileur installe, et en installer un pour une premiere
# approche serait disproportionne. On decompose donc par difference, avec ce
# que le moteur sait deja faire :
#
#   perft            generation de coups + do_move, RIEN d'autre
#   bench sans reseau  recherche complete, evaluation PSQT (quelques additions)
#   bench avec reseau  recherche complete, evaluation NNUE
#
# L'ecart entre les deux derniers donne le cout de l'evaluation par noeud ;
# l'ecart au perft donne le cout de tout le reste -- tri des coups, sondes de
# table, historiques.
#
# On fait aussi varier la taille de la table de transposition : si le NPS
# s'effondre quand elle depasse le cache L3 (12 Mo ici), c'est que les defauts
# de cache dominent, et alors il faut optimiser la memoire, pas le calcul.
set -u
cd /c/Onechess

REPEATS=${REPEATS:-3}
EXE=engine/bin/onechess.exe
NET=C:/Onechess/engine/bin/onechess.nnue

libre() {
    local n
    n=$(powershell.exe -NoProfile -Command "@(Get-Process onechess,stockfish-windows-x86-64-avx2 -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r')
    [ "${n:-0}" -eq 0 ]
}
if ! libre; then
    echo "REFUS : des moteurs tournent deja." >&2; exit 1
fi

median() { sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }

bench_nps() {   # $1 = options supplementaires
    for _ in $(seq 1 "$REPEATS"); do
        printf '%s\nisready\nbench 12\nquit\n' "$1" | "$EXE" 2>&1 | awk '/^NPS/ {print $3}'
    done | median
}

echo "=== decomposition du temps  $(date '+%H:%M')  ($REPEATS repetitions) ==="
echo

nps_nnue=$(bench_nps "setoption name EvalFile value $NET")
nps_psqt=$(bench_nps "setoption name EvalFile value <empty>")

echo "--- 1. cout de l'evaluation ---"
printf "  recherche + NNUE : %'d NPS   soit %.0f ns/noeud\n" "$nps_nnue" "$(awk -v n=$nps_nnue 'BEGIN{print 1e9/n}')"
printf "  recherche + PSQT : %'d NPS   soit %.0f ns/noeud\n" "$nps_psqt" "$(awk -v n=$nps_psqt 'BEGIN{print 1e9/n}')"
awk -v a="$nps_nnue" -v b="$nps_psqt" 'BEGIN {
    d = 1e9/a - 1e9/b
    printf "  surcout NNUE : %.0f ns/noeud, soit %.0f %% du budget\n", d, 100*d*a/1e9 }'

echo
echo "--- 2. cout de la generation de coups seule ---"
perft=$(for _ in $(seq 1 "$REPEATS"); do
    printf 'position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1
perft 5
quit
'       | "$EXE" 2>&1 | awk '/knps/ {print $(NF-1) * 1000}'
done | median)
printf "  perft : %'d noeuds/s   soit %.1f ns/noeud
" "$perft" "$(awk -v n=$perft 'BEGIN{print 1e9/n}')"
awk -v p="$perft" -v s="$nps_nnue" 'BEGIN {
    printf "  un noeud de recherche coute %.0fx un noeud de perft
", (1e9/s)/(1e9/p)
    printf "  -> generation de coups et do_move : environ %.1f %% du temps
", 100*(1e9/p)/(1e9/s) }'

echo
echo "--- 3. le cache est-il le mur ? (L3 = 12 Mo partages) ---"
# ATTENTION : ne PAS utiliser `bench` ici. Il efface la table avant chacune de
# ses 20 positions ; a 2 Go cela fait 40 Go de memset comptes dans le temps, et
# la premiere version de ce script a ainsi annonce un NPS divise par 11 -- une
# illusion qui aurait fait conclure a une recherche etranglee par la memoire.
# On fait donc UN seul effacement, puis une recherche longue.
# Pas de "quit" : la recherche est asynchrone et s'arreterait aussitot ; c'est
# la fin d'entree qui la laisse aller au bout.
FEN="r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10"
search_nps() {   # $1 = hash en Mo
    for _ in $(seq 1 "$REPEATS"); do
        printf 'setoption name Hash value %s\nsetoption name EvalFile value %s\nucinewgame\nisready\nposition fen %s\ngo nodes 8000000\n' \
               "$1" "$NET" "$FEN" | "$EXE" 2>&1 | grep '^info depth' | tail -1 \
          | grep -o 'nps [0-9]*' | awk '{print $2}'
    done | median
}
printf "  %-14s %14s %10s\n" "hash" "NPS median" "vs 16 Mo"
base=""
for h in 16 64 128 256 512 1024; do
    n=$(search_nps "$h")
    [ -z "$base" ] && base="$n"
    awk -v h="$h" -v n="$n" -v b="$base" 'BEGIN {printf "  %-14s %14d %9.2fx\n", h" Mo", n, n/b}'
done
echo
echo "Mesure du 10-09 : -12 % seulement entre 16 Mo et 1 Go. La table de"
echo "transposition n'est PAS le goulot de ce moteur."
