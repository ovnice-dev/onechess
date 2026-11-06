#!/usr/bin/env bash
# Phase 10, A3 : le reseau a penalite (lambda 0,005) mene a 20 epoques, reprise
# du point de l'epoque 3. Ensuite : ftperm, puis match au temps contre R.
cd /c/Onechess
python -u tools/train.py --l1 16 --l2 32 --pairmul 1 --stream "C:/Onechess/data/r_n5/*.bin" \
  --chunk-files 10 --epochs 20 --buckets mat4 --out-buckets 4 --act-l1 0.005 \
  --out reseau/essais/local/a3_l0.005.nnue --checkpoint reseau/essais/local/a3_l0.005.ckpt \
  >> reseau/essais/local/a3_l0.005.log 2>&1
