# Entrainement continu, semaine 1, sur le TPU de Kaggle : poids de depart = meilleure epoque du final
# (sortie du noyau onechess-finalb-tpu), tranche NEUVE de nodes5000pv2_UHO (octets 7 a 14 Go).
# Ne pas importer jax ici : le processus reserverait le TPU et l'entraineur ne pourrait plus y acceder.
import glob, io, os, subprocess, sys, tarfile, time, urllib.request

W = '/kaggle/working'
t0 = time.time()


def lancer(cmd):
    """Affiche en direct et recopie dans un fichier conserve par Kaggle ; rend le code de sortie."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(W + '/hebdo1.log', 'a', buffering=1) as j:
        for l in p.stdout:
            print(l, end='', flush=True)
            j.write(l)
    return p.wait()


# Garde-fou : jamais d'entrainement sur processeur (dans un processus a part : il libere le TPU en sortant)
_d = subprocess.run([sys.executable, '-c', 'import jax; print(jax.devices()[0].platform, jax.device_count())'],
                    capture_output=True, text=True).stdout.strip().splitlines()
print('machine :', _d, flush=True)
if not _d or not _d[-1].startswith('tpu'):
    sys.exit('PAS DE TPU : arret')

tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(
    'https://codeload.github.com/ovnice-dev/onechess-tools/tar.gz/main').read())).extractall('/tmp/code')
TOOLS = glob.glob('/tmp/code/*/tools')[0]
for mod, pip in (('optax', ['optax']), ('torch', ['torch', '--index-url', 'https://download.pytorch.org/whl/cpu'])):
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pip, check=True)

# Donnees : exactement celles du final sur GPU (7 premiers Go, memes graines, 4 parts, meme validation)
if lancer([sys.executable, '-u', TOOLS + '/hf_sources.py', '--url',
           'https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/nodes5000pv2_UHO.binpack',
           '--prefixe', 'n5s1', '--dest', '/tmp/donnees', '--debut', '7000000000', '--octets', '7000000000',
           '--garde', '1.0', '--fixe', '240', '--graine', '12', '--morceau', '4000000000', '--parts', '4']):
    sys.exit('preparation des donnees en echec')
import numpy as np
v = sorted(glob.glob('/tmp/donnees/*_p00_*.bin'))[0]
os.rename(v, '/tmp/donnees/z_val.bin')
np.fromfile('/tmp/donnees/z_val.bin', dtype='uint8').reshape(-1, 32)[:5_000_000].tofile('/tmp/donnees/z_val.bin')
print(f"donnees pretes : {time.time() - t0:.0f} s, {len(glob.glob('/tmp/donnees/*.bin'))} fichiers", flush=True)

COMMUN = [sys.executable, '-u', TOOLS + '/train_jax.py', '--ft-dense', '1', '--stream', '/tmp/donnees/*.bin',
          '--hidden', '1024', '--buckets', 'mat4', '--out-buckets', '4', '--chunk-files', '10']
# Essai court : verifie le TPU (8 puces) et donne la vitesse avant l'entrainement
if lancer(COMMUN + ['--max-pas', '300', '--val', '100000']):
    sys.exit("essai de vitesse en echec : voir le message ci-dessus")

import torch
ck = glob.glob('/kaggle/input/**/final_tpu.ckpt', recursive=True)
if not ck:
    sys.exit('final_tpu.ckpt introuvable dans /kaggle/input')
c = torch.load(ck[0], map_location='cpu', weights_only=False)
torch.save(c.get('best_state') or c['model'], '/tmp/init.pt')
print('poids de depart :', ck[0], 'epoque', c.get('best', ['?'] * 3)[2], flush=True)
# Pas reduit (2e-4, cosinus jusqu'a 2e-4/30) ; 60 epoques de 300 M ; arret propre apres 7 h.
code = lancer(COMMUN + ['--epochs', '60', '--epoch-files', '60', '--init', '/tmp/init.pt', '--lr', '2e-4',
                        '--limite-s', str(7 * 3600), '--checkpoint', W + '/hebdo1.pkl',
                        '--ckpt-torch', W + '/hebdo1.ckpt', '--out', W + '/hebdo1.nnue', '--save-pt', W + '/hebdo1.pt'])
print('code', code, f'{time.time() - t0:.0f} s', flush=True)
