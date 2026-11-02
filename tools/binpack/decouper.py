"""Decoupe un binpack en K parts sur des frontieres de blocs.

Un binpack est une suite de blocs independants : 'BINP', taille (uint32 LE),
puis les donnees. Chaque part est donc un binpack valide que
binpack_vers_bin convertit seul : K conversions en parallele. Un dernier bloc
incomplet (fichier tronque par un telechargement partiel) est ignore.

    python decouper.py fichier.binpack prefixe_part K
"""
import os
import struct
import sys


def frontieres(chemin):
    """Positions de debut de chaque bloc complet, et la fin du dernier."""
    taille = os.path.getsize(chemin)
    debuts = []
    with open(chemin, 'rb') as f:
        pos = 0
        while pos + 8 <= taille:
            f.seek(pos)
            h = f.read(8)
            if h[:4] != b'BINP':
                raise SystemExit(f"bloc invalide a l'octet {pos}")
            n = struct.unpack('<I', h[4:])[0]
            if pos + 8 + n > taille:
                break  # bloc tronque
            debuts.append(pos)
            pos += 8 + n
    return debuts, pos


def main():
    src, prefixe, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    debuts, fin = frontieres(src)
    if not debuts:
        raise SystemExit("aucun bloc complet")
    k = min(k, len(debuts))
    coupes = [debuts[len(debuts) * i // k] for i in range(k)] + [fin]
    with open(src, 'rb') as f:
        for i in range(k):
            f.seek(coupes[i])
            reste = coupes[i + 1] - coupes[i]
            with open(f"{prefixe}{i:02d}.binpack", 'wb') as o:
                while reste:
                    b = f.read(min(reste, 1 << 26))
                    o.write(b)
                    reste -= len(b)
    print(f"{len(debuts)} blocs, {fin / 1e9:.2f} Go utiles, {k} parts", flush=True)


if __name__ == '__main__':
    main()
