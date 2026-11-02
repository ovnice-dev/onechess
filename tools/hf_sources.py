"""Prepare des positions publiques (binpacks Hugging Face) pour train.py.

Sur Kaggle (Internet active) : telecharge un binpack -- en entier ou ses N
premiers octets --, le decoupe en parts sur des frontieres de blocs, convertit
les parts en parallele (binpack_vers_bin, compile au besoin) en fichiers de
32 octets par position dans --dest, puis efface le binpack. Tout est
deterministe (memes octets, memes graines) : refait a chaque session, le
resultat est identique, donc la reprise d'un entrainement reste exacte.

    python hf_sources.py --url URL --prefixe n5 --dest /tmp/donnees \
        [--octets 0] [--garde 1.0] [--fixe 0] [--parts 4] [--graine 1]

Echelle : --fixe 240 pour l'autojeu Stockfish nodes5000 (mesure du 1-11) ;
sans --fixe, a(materiel) de Stockfish 19 (relabel Leela BT4).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ICI = os.path.dirname(os.path.abspath(__file__))


def convertisseur():
    exe = os.path.join('/tmp', 'binpack_vers_bin')
    if not os.path.exists(exe):
        src = os.path.join(ICI, 'binpack', 'binpack_vers_bin.cpp')
        subprocess.run(['g++', '-O2', '-std=c++20', '-march=native', '-o', exe, src], check=True)
    return exe


def telecharger(url, dest, octets):
    req = urllib.request.Request(url)
    if octets:
        req.add_header('Range', f'bytes=0-{octets - 1}')
    t, n = time.time(), 0
    with urllib.request.urlopen(req) as r, open(dest, 'wb') as f:
        while True:
            b = r.read(1 << 24)
            if not b:
                break
            f.write(b)
            n += len(b)
    print(f"telecharge {n / 1e9:.2f} Go en {time.time() - t:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--prefixe', required=True)
    ap.add_argument('--dest', default='/tmp/donnees')
    ap.add_argument('--octets', type=int, default=0)
    ap.add_argument('--garde', type=float, default=1.0)
    ap.add_argument('--fixe', type=float, default=0)
    ap.add_argument('--parts', type=int, default=4)
    ap.add_argument('--graine', type=int, default=1)
    ap.add_argument('--par', type=int, default=5000000)
    a = ap.parse_args()

    os.makedirs(a.dest, exist_ok=True)
    tmp = os.path.join('/tmp', f'src_{a.prefixe}')
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(tmp, 'source.binpack')
    telecharger(a.url, src, a.octets)

    t = time.time()
    subprocess.run([sys.executable, os.path.join(ICI, 'binpack', 'decouper.py'), src,
                    os.path.join(tmp, 'part'), str(a.parts)], check=True)
    os.remove(src)
    exe = convertisseur()
    procs = []
    for i, p in enumerate(sorted(glob.glob(os.path.join(tmp, 'part*.binpack')))):
        cmd = [exe, p, os.path.join(a.dest, f'{a.prefixe}_p{i:02d}'), '--garde', str(a.garde),
               '--graine', str(a.graine * 100 + i), '--par', str(a.par)]
        if a.fixe:
            cmd += ['--fixe', str(a.fixe)]
        procs.append(subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True))
    for p in procs:
        err = p.communicate()[1]
        print(err.strip().splitlines()[-1] if err.strip() else '?', flush=True)
        if p.returncode:
            sys.exit(f"conversion en echec (code {p.returncode})")
    shutil.rmtree(tmp)
    n = sum(os.path.getsize(f) for f in glob.glob(os.path.join(a.dest, f'{a.prefixe}_p*.bin'))) // 32
    print(f"{a.prefixe} : {n:,} positions dans {a.dest} ({time.time() - t:.0f} s de conversion)",
          flush=True)


if __name__ == '__main__':
    main()
