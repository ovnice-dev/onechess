#!/usr/bin/env bash
# Range les morceaux de SPRT joues sur Kaggle (noyaux ci/kaggle_nuit/m-*) dans
# ci_resultats/lot/<test>/kaggle/, ou le bilan du lot les additionne.
#   bash ci/recolter_kaggle_lot.sh
K="/c/Users/SMILE/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/LocalCache/local-packages/Python313/Scripts/kaggle.exe"
cd /c/Onechess
for d in ci/kaggle_nuit/m-*/; do
    id=$(python -c "import json;print(json.load(open('$d/kernel-metadata.json'))['id'])")
    st=$("$K" kernels status "$id" 2>&1 | grep -o 'Status\.[A-Z]*')
    [ "$st" = "Status.COMPLETE" ] || { echo "$id : ${st:-pas lance}"; continue; }
    out="_kaggle/sorties/${id#ovnice/}"
    "$K" kernels output "$id" -p "$out" >/dev/null 2>&1
    for s in "$out"/*.state; do
        [ -f "$s" ] || continue
        nom=$(basename "$s"); nom=${nom%%-kaggle-*}
        mkdir -p "ci_resultats/lot/$nom/kaggle"
        cp "$s" "ci_resultats/lot/$nom/kaggle/"
        echo "$id -> ci_resultats/lot/$nom/kaggle/$(basename "$s")"
    done
done
