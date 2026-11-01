#!/usr/bin/env bash
# Q (moitie l150, moitie publique, validation = celle de P) apres P.
cd /c/Onechess
until grep -q "reseau ecrit" reseau/essais/local/p.log; do sleep 120; done
mkdir -p data/q_mix
for f in $(ls data/l150/*.bin | head -20); do ln "$f" data/q_mix/m_l150_$(basename $f); done
for f in $(ls data/publics/p_fish/*.bin | head -17); do ln "$f" data/q_mix/m_pub_$(basename $f); done
ln data/publics/p_fish/fish_0033.bin data/q_mix/z_val_fish_0033.bin
python -u tools/train.py --stream "C:/Onechess/data/q_mix/*.bin" --chunk-files 10 --epochs 20 --buckets mat4 --out-buckets 4 --out reseau/essais/local/q_mix_mat4_o4.nnue --checkpoint reseau/essais/local/q.ckpt > reseau/essais/local/q.log 2>&1
