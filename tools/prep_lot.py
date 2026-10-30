#!/usr/bin/env python3
"""Prepare un gros lot d'entrainement NNUE pour train.py --stream.

Au-dela de ~300 M de positions, le lot ne tient plus ni en memoire vive
(train.py charge tout, plus une copie pour le melange) ni sur la carte (32 o
par position : 500 M = 16 Go, la taille d'une T4). On le prepare donc une
fois pour toutes, sur disque :

  1. doublons : une cle de 8 octets par position, calculee fichier par
     fichier (les enregistrements ne sont jamais tous en memoire) ; on garde
     les k premiers exemplaires de chaque position, comme dedup_recs ;
  2. melange : chaque position gardee part dans un des B fichiers de sortie,
     tire au hasard ; puis chaque fichier est melange en memoire. Le resultat
     est un melange uniforme, en lectures et ecritures sequentielles, sans
     jamais plus d'un fichier de sortie en memoire.

--marque MOTIF (repetable) : les positions des fichiers qui y correspondent
recoivent 1 dans l'octet 29 (premier octet de bourrage, toujours nul dans nos
donnees). train.py --lambda-auto applique alors une autre cible a ces
positions (l'autojeu, dont les resultats sont fiables, contrairement aux
parties humaines de Lichess). Les fichiers sources ne sont jamais modifies.

Chaque fichier de sortie est donc un echantillon au hasard du lot entier :
train.py --stream en charge quelques-uns a la fois sur la carte, et garde le
dernier pour la validation.

    python tools/prep_lot.py --data "data/nnue_elite/**/*.bin" --data "data/autojeu/*.bin" \
        --dedup 4 --out data/lot_stream --files 64
"""

import argparse
import glob
import os
import time

import numpy as np

RECORD = 32


def lister(motifs):
    files = []
    for m in motifs:
        if m.startswith("@"):
            files += [l.strip() for l in open(m[1:], encoding="utf-8") if l.strip()]
        else:
            files += glob.glob(m, recursive=True)
    files = sorted(set(files))
    if not files:
        raise SystemExit("aucun fichier d'entree")
    return files


def cles(r):
    """Meme hachage que dedup_recs de train.py : plateau (24 o) + trait."""
    q = np.ascontiguousarray(r[:, :24]).view("<u8")
    with np.errstate(over="ignore"):
        h = q[:, 0] * np.uint64(0x9E3779B97F4A7C15)
        h ^= (q[:, 1] + np.uint64(0x632BE59BD9B4E019)) * np.uint64(0xC2B2AE3D27D4EB4F)
        h ^= (q[:, 2] + np.uint64(0x165667B19E3779F9)) * np.uint64(0x94D049BB133111EB)
        h ^= r[:, 26].astype(np.uint64) * np.uint64(0xD6E8FEB86659FD93)
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True,
                    help="motif glob (** permis) ou @liste.txt ; repetable")
    ap.add_argument("--out", required=True, help="dossier de sortie (cree)")
    ap.add_argument("--files", type=int, default=64)
    ap.add_argument("--dedup", type=int, default=4, help="0 = garder tout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--marque", action="append", default=[],
                    help="motif glob : ces fichiers recoivent 1 dans l'octet 29")
    ap.add_argument("--tranche", type=int, default=1 << 22,
                    help="positions lues a la fois")
    args = ap.parse_args()

    t0 = time.time()
    files = lister(args.data)
    tailles = [os.path.getsize(f) // RECORD for f in files]
    n = sum(tailles)
    print(f"{n:,} positions dans {len(files)} fichiers", flush=True)
    marques = set(lister(args.marque)) if args.marque else set()
    if marques:
        print(f"{len(marques & set(files))} fichiers marques (octet 29 = 1)", flush=True)

    # ---------------------------------------------------- 1. doublons
    garde = np.ones(n, dtype=bool)
    if args.dedup:
        key = np.empty(n, dtype=np.uint64)
        pos = 0
        for f, k in zip(files, tailles):
            mm = np.memmap(f, dtype=np.uint8, mode="r", shape=(k, RECORD))
            for a in range(0, k, args.tranche):
                r = np.asarray(mm[a:a + args.tranche])
                key[pos:pos + len(r)] = cles(r)
                pos += len(r)
            del mm
        order = np.argsort(key, kind="stable")
        key = key[order]
        debut = np.empty(n, dtype=bool)
        debut[0] = True
        np.not_equal(key[1:], key[:-1], out=debut[1:])
        del key
        it = np.int32 if n < 2**31 else np.int64   # 2 Go au lieu de 4 a 500 M
        rang = np.arange(n, dtype=it)
        tete = np.where(debut, rang, 0).astype(it)
        del debut
        np.maximum.accumulate(tete, out=tete)
        rang -= tete
        del tete
        garde[:] = False
        garde[order[rang < args.dedup]] = True
        del order, rang
    g = int(garde.sum())
    print(f"doublons : {n - g:,} retirees (au plus {args.dedup} exemplaires), "
          f"{g:,} gardees  ({time.time() - t0:.0f} s)", flush=True)

    # ---------------------------------------------------- 2. repartition
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    noms = [os.path.join(args.out, f"lot_{i:03d}.bin") for i in range(args.files)]
    sorties = [open(nm, "wb") for nm in noms]
    pos = 0
    for f, k in zip(files, tailles):
        mm = np.memmap(f, dtype=np.uint8, mode="r", shape=(k, RECORD))
        for a in range(0, k, args.tranche):
            r = np.asarray(mm[a:a + args.tranche])
            r = r[garde[pos:pos + len(r)]]
            pos += min(args.tranche, k - a)
            if f in marques:
                assert not r[:, 29].any(), f"octet 29 deja utilise : {f}"
                r[:, 29] = 1
            dest = rng.integers(0, args.files, size=len(r))
            idx = np.argsort(dest, kind="stable")
            r, dest = r[idx], dest[idx]
            bornes = np.searchsorted(dest, np.arange(args.files + 1))
            for i in range(args.files):
                if bornes[i + 1] > bornes[i]:
                    sorties[i].write(r[bornes[i]:bornes[i + 1]].tobytes())
        del mm
    for s in sorties:
        s.close()
    del garde
    print(f"repartition en {args.files} fichiers  ({time.time() - t0:.0f} s)", flush=True)

    # ---------------------------------------------------- 3. melange interne
    total = 0
    for nm in noms:
        r = np.fromfile(nm, dtype=np.uint8).reshape(-1, RECORD)
        r = r[rng.permutation(len(r))]
        r.tofile(nm)
        total += len(r)
    assert total == g, (total, g)
    if marques:
        nm_ = sum(int(np.fromfile(nm, dtype=np.uint8).reshape(-1, RECORD)[:, 29].sum()) for nm in noms)
        print(f"positions marquees : {nm_:,} sur {total:,}", flush=True)
    print(f"termine : {total:,} positions, {args.files} fichiers de ~{total // args.files:,}, "
          f"{total * RECORD / 1e9:.2f} Go  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
