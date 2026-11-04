#!/usr/bin/env bash
# Binaires de diffusion d'Onechess (Windows, statiques, PGO), un par famille de
# processeurs, dans engine/bin/diffusion/ :
#   onechess-bmi2.exe   AVX2 + PEXT rapide (Intel depuis Haswell, AMD Zen 3 et suivants)
#   onechess-avx2.exe   AVX2 sans PEXT (AMD Zen 1 et 2, ou PEXT est tres lent)
#   onechess-sse41.exe  sans AVX2 (vieux processeurs ; reseau en scalaire, lent)
# Chaque binaire doit donner la signature de reference (ci/signature.txt).
#   bash tools/diffusion.sh
set -e
cd "$(dirname "$0")/../engine"
mkdir -p bin/diffusion
# le reseau voyage a cote des executables (le moteur le cherche la)
cp bin/onechess.nnue bin/diffusion/onechess.nnue
attendu=$(cat ../ci/signature.txt)
for cible in "bmi2|-march=x86-64-v3 -mtune=generic -DUSE_PEXT" \
             "avx2|-march=x86-64-v3 -mtune=generic" \
             "sse41|-march=x86-64-v2 -mtune=generic"; do
    nom=${cible%%|*}; arch=${cible#*|}
    ARCH="$arch" OUT="bin/diffusion/onechess-$nom.exe" ./build.sh pgo > "bin/diffusion/$nom.log" 2>&1
    n=$(printf 'bench 12\nquit\n' | "./bin/diffusion/onechess-$nom.exe" | grep "Noeuds totaux" | awk '{print $4}')
    nps=$(printf 'bench 12\nquit\n' | "./bin/diffusion/onechess-$nom.exe" | grep "^NPS" | awk '{print $3}')
    [ "$n" = "$attendu" ] && etat=OK || etat="SIGNATURE DIFFERENTE ($n)"
    echo "onechess-$nom.exe : $etat, $nps NPS"
done
