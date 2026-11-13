# Fin du reseau final sur le TPU de Kaggle (session lancee depuis la page web : l'API ne donne pas de TPU).
# Reprend le point GPU (pre-release final-reprise) ; sorties dans /kaggle/working.
# Ne pas importer jax ici : le processus reserverait le TPU et l'entraineur ne pourrait plus y acceder.
import glob, io, os, subprocess, sys, tarfile, time, urllib.request

W = '/kaggle/working'
t0 = time.time()


def lancer(cmd):
    """Affiche en direct et recopie dans un fichier conserve par Kaggle ; rend le code de sortie."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(W + '/final_tpu.log', 'a', buffering=1) as j:
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
           '--prefixe', 'n5', '--dest', '/tmp/donnees', '--octets', '7000000000', '--garde', '1.0', '--fixe', '240',
           '--graine', '11', '--morceau', '4000000000', '--parts', '4']):
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

REPRISE = '/tmp/final_gpu.ckpt'
urllib.request.urlretrieve('https://github.com/ovnice-dev/onechess/releases/download/final-reprise/final.ckpt', REPRISE)
# Limite des sessions TPU : 9 h. Arret propre apres 8 h d'entrainement ; a reprendre avec final_tpu.ckpt.
code = lancer(COMMUN + ['--epochs', '60', '--epoch-files', '60', '--reprise-torch', REPRISE,
                        '--limite-s', str(8 * 3600), '--checkpoint', W + '/final_tpu.pkl',
                        '--ckpt-torch', W + '/final_tpu.ckpt', '--out', W + '/final.nnue', '--save-pt', W + '/final.pt'])
print('code', code, f'{time.time() - t0:.0f} s', flush=True)
