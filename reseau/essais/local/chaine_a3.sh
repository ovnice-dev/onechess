#!/usr/bin/env bash
# Phase 10, A3 : parcimonie des couches cachees. 512 + l1 16/32 + produit par
# paires (comme RHP), donnees de R, 3 epoques sur 20 (meme programme) :
# sans penalite, puis avec --act-l1. Comparer erreur et zeros (nnz512.exe).
cd /c/Onechess
for l in 0 0.005; do
  python -u tools/train.py --l1 16 --l2 32 --pairmul 1 --stream "C:/Onechess/data/r_n5/*.bin" \
    --chunk-files 10 --epochs 20 --stop-after 3 --buckets mat4 --out-buckets 4 --act-l1 $l \
    --out reseau/essais/local/a3_l$l.nnue --checkpoint reseau/essais/local/a3_l$l.ckpt \
    > reseau/essais/local/a3_l$l.log 2>&1
done
