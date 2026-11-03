#!/usr/bin/env python3
"""Reseaux synthetiques au format du moteur (couches cachees, pairmul,
paires de pions) pour comparer l'inference C++ a tools/check_nnue.py."""
import struct
import sys

import numpy as np

MAGIC4 = 0x344E4E4F
PAIRS_FLAG, HIDDEN_FLAG, PAIRMUL_FLAG = 256, 1024, 2048
PAIR_INPUTS = 96 * 95 // 2


def write(path, nb=4, mirror=2, ob=4, l1=32, l2=32, pairmul=False, pairs=False, seed=1, hl=512):
    rng = np.random.default_rng(seed)
    act = HIDDEN_FLAG | (PAIRMUL_FLAG if pairmul else 0) | (PAIRS_FLAG if pairs else 0)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIIIII", MAGIC4, hl, nb, mirror, ob, act))
        layout = np.zeros(64, dtype=np.uint8) if mirror == 2 else (np.arange(64) % nb).astype(np.uint8)
        f.write(layout.tobytes())
        fw = rng.normal(0, 30, size=(nb * 768, hl)).round().clip(-300, 300).astype("<i2")
        fb = rng.normal(60, 40, size=hl).round().astype("<i2")
        f.write(fw.tobytes())
        f.write(fb.tobytes())
        if pairs:
            pw = rng.normal(0, 6, size=(PAIR_INPUTS, hl)).round().astype("<i2")
            f.write(pw.tobytes())
        f.write(struct.pack("<II", l1, l2))
        if pairmul:
            w1 = rng.integers(-6, 7, size=(ob * l1, hl)).astype("i1")
            b1 = rng.normal(500, 2000, size=ob * l1).round().astype("<i4")
        else:
            w1 = rng.integers(-8, 9, size=(ob * l1, 2 * hl)).astype("<i2")
            b1 = rng.normal(0, 30000, size=ob * l1).round().astype("<i4")
        w2 = rng.integers(-40, 41, size=(ob * l2, l1)).astype("<i2")
        b2 = rng.normal(0, 3000, size=ob * l2).round().astype("<i4")
        w3 = rng.integers(-126, 127, size=(ob, l2)).astype("<i2")
        b3 = rng.normal(0, 10000, size=ob).round().astype("<i4")
        for a in (w1, b1, w2, b2, w3, b3):
            f.write(a.tobytes())


if __name__ == "__main__":
    out = sys.argv[1]
    write(out + "_h32.nnue", l1=32, l2=32)
    write(out + "_h16_ob8.nnue", ob=8, l1=16, l2=8, seed=2)
    write(out + "_pm32.nnue", l1=32, l2=32, pairmul=True, seed=3)
    write(out + "_pm64_ob8.nnue", ob=8, l1=64, l2=16, pairmul=True, seed=4)
    write(out + "_pm16_k16.nnue", nb=16, mirror=1, ob=4, l1=16, l2=32, pairmul=True, seed=5)
    write(out + "_h32_pairs.nnue", nb=1, mirror=0, ob=4, l1=32, l2=16, pairs=True, seed=6)
    print("ok")
