# Phase 10, A5 : greffe des entrees "menaces" sur le reseau final (GPU T4 de Kaggle).
# Poids des menaces neufs a zero (le reseau part identique au final), puis affinage
# sur les memes donnees que le final (7 premiers Go). Juge : match au temps contre reseau-1.
import glob, io, os, subprocess, sys, tarfile, time, urllib.request
W = '/kaggle/working'
t0 = time.time()
def lancer(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(W + '/greffe.log', 'a', buffering=1) as j:
        for l in p.stdout:
            print(l, end='', flush=True); j.write(l)
    return p.wait()
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(
    'https://codeload.github.com/ovnice-dev/onechess/tar.gz/main').read())).extractall('/tmp/code')
TOOLS = glob.glob('/tmp/code/*/tools')[0]
ck = glob.glob('/kaggle/input/**/final_tpu.ckpt', recursive=True)
if not ck:
    sys.exit("final_tpu.ckpt introuvable dans /kaggle/input")
if lancer([sys.executable, '-u', TOOLS + '/hf_sources.py', '--url',
           'https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/nodes5000pv2_UHO.binpack',
           '--prefixe', 'n5', '--dest', '/tmp/donnees', '--octets', '7000000000', '--garde', '1.0', '--fixe', '240',
           '--graine', '11', '--morceau', '4000000000', '--parts', '4']):
    sys.exit('preparation des donnees en echec')
import numpy as np
v = sorted(glob.glob('/tmp/donnees/*_p00_*.bin'))[0]
os.rename(v, '/tmp/donnees/z_val.bin')
np.fromfile('/tmp/donnees/z_val.bin', dtype='uint8').reshape(-1, 32)[:5_000_000].tofile('/tmp/donnees/z_val.bin')
print(f"donnees pretes : {time.time() - t0:.0f} s", flush=True)
# Quota GPU restant ~7 h 20 au lancement : 8 epoques (~40 min chacune avec les menaces) + marge ;
# coupure de securite a 6 h 30 de session (le point de reprise reste dans la sortie).
reste = int(6.5 * 3600 - (time.time() - t0))
code = lancer(['timeout', str(reste), sys.executable, '-u', TOOLS + '/train.py', '--stream', '/tmp/donnees/*.bin',
               '--chunk-files', '10', '--epoch-files', '60', '--hidden', '1024', '--buckets', 'mat4',
               '--out-buckets', '4', '--threats', '1', '--greffe', ck[0], '--epochs', '8', '--lr', '3e-4',
               '--out', W + '/menaces.nnue', '--checkpoint', W + '/menaces.ckpt', '--save-pt', W + '/menaces.pt'])
print('code', code, f'{time.time() - t0:.0f} s', flush=True)
