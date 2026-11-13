# Phase 10, C : quelles menaces valent leur cout ? Session TPU de Kaggle (v5e-8).
# Quatre affinages du reseau final, A ENTRAINEMENT EGAL (memes donnees, memes epoques, meme pas) :
#   temoin       : sans menaces (mesure ce que donne l'entrainement en plus, a retrancher)
#   pleines      : menaces completes, 1024 neurones (version Stockfish, 61 Mo dans le moteur)
#   etroit       : bloc de 256 neurones (15 Mo : tient dans le cache du processeur)
#   etroit-enn.  : 256 neurones, sans les defenses (~4 fois moins de lignes a lire)
# Juge ici : l'erreur de validation (meme fichier pour tous). Le moteur ne sera adapte qu'a la
# variante qui garde l'essentiel du gain ; puis vitesse et match au temps.
# Ne pas importer jax ici : le processus reserverait le TPU.
import glob, io, os, re, subprocess, sys, tarfile, time, urllib.request

W = '/kaggle/working'
t0 = time.time()


def lancer(cmd, journal):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sortie = []
    with open(W + '/' + journal, 'a', buffering=1) as j:
        for l in p.stdout:
            print(l, end='', flush=True)
            j.write(l)
            sortie.append(l)
    return p.wait(), ''.join(sortie)


_d = subprocess.run([sys.executable, '-c', 'import jax; print(jax.devices()[0].platform, jax.device_count())'],
                    capture_output=True, text=True).stdout.strip().splitlines()
print('machine :', _d, flush=True)
if not _d or not _d[-1].startswith('tpu'):
    sys.exit('PAS DE TPU : arret')

tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(
    'https://codeload.github.com/ovnice-dev/onechess/tar.gz/main').read())).extractall('/tmp/code')
TOOLS = glob.glob('/tmp/code/*/tools')[0]
for mod, pip in (('optax', ['optax']), ('torch', ['torch', '--index-url', 'https://download.pytorch.org/whl/cpu'])):
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pip, check=True)

# Donnees : celles du final (le reseau les connait ; seule change l'architecture)
code, _ = lancer([sys.executable, '-u', TOOLS + '/hf_sources.py', '--url',
                  'https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/nodes5000pv2_UHO.binpack',
                  '--prefixe', 'n5', '--dest', '/tmp/donnees', '--octets', '7000000000', '--garde', '1.0', '--fixe', '240',
                  '--graine', '11', '--morceau', '4000000000', '--parts', '4'], 'menaces.log')
if code:
    sys.exit('preparation des donnees en echec')
import numpy as np
v = sorted(glob.glob('/tmp/donnees/*_p00_*.bin'))[0]
os.rename(v, '/tmp/donnees/z_val.bin')
np.fromfile('/tmp/donnees/z_val.bin', dtype='uint8').reshape(-1, 32)[:2_000_000].tofile('/tmp/donnees/z_val.bin')
print(f"donnees pretes : {time.time() - t0:.0f} s", flush=True)

import torch
ck = glob.glob('/kaggle/input/**/final_tpu.ckpt', recursive=True)
if not ck:
    sys.exit('final_tpu.ckpt introuvable dans /kaggle/input')
c = torch.load(ck[0], map_location='cpu', weights_only=False)
torch.save(c.get('best_state') or c['model'], '/tmp/init.pt')

VARIANTES = [('temoin', []),
             ('etroit_ennemis', ['--menaces', '1', '--menaces-largeur', '256', '--menaces-ennemis', '1']),
             ('etroit', ['--menaces', '1', '--menaces-largeur', '256']),
             ('pleines', ['--menaces', '1'])]
COMMUN = [sys.executable, '-u', TOOLS + '/train_jax.py', '--stream', '/tmp/donnees/*.bin', '--hidden', '1024',
          '--buckets', 'mat4', '--out-buckets', '4', '--chunk-files', '10', '--epoch-files', '60',
          '--init', '/tmp/init.pt', '--lr', '3e-4', '--val', '2000000']

# 1. Vitesse de chaque variante (300 pas), pour fixer un nombre d'epoques commun qui tienne
vitesse = {}
for nom, opts in VARIANTES:
    dense = ['--ft-dense', '1']
    _, out = lancer(COMMUN + dense + opts + ['--max-pas', '300'], 'menaces.log')
    m = re.search(r'= ([\d,]+) positions/s', out)
    vitesse[nom] = int(m.group(1).replace(',', '')) if m else 0
    print(f"VITESSE {nom} : {vitesse[nom]:,} positions/s", flush=True)
plus_lente = min(v for v in vitesse.values() if v > 0)
budget = 80 * 60   # par variante (session TPU limitee a 9 h)
epoques = max(2, min(12, int(budget * plus_lente / 300e6)))
print(f"EPOQUES par variante : {epoques} (variante la plus lente : {plus_lente:,} positions/s)", flush=True)

# 2. Affinages
bilan = {}
for nom, opts in VARIANTES:
    dense = ['--ft-dense', '1']
    _, out = lancer(COMMUN + dense + opts + ['--epochs', str(epoques), '--limite-s', str(int(budget * 1.5)),
                                             '--out', f'{W}/{nom}.nnue', '--checkpoint', f'{W}/{nom}.pkl'],
                    f'{nom}.log')
    cp = re.findall(r'erreur\s+([\d.]+) cp', out)
    bilan[nom] = cp[-1] if cp else '?'
    print(f"BILAN {nom} : {bilan[nom]} cp", flush=True)
    for f in glob.glob(f'{W}/{nom}.pkl'):
        os.remove(f)   # 0,7 Go par variante : seul le reseau compte
print('RESUME :', bilan, f'{time.time() - t0:.0f} s', flush=True)
