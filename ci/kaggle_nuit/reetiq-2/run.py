# Reetiquetage (essai 9.2) : une partie de l'echantillon Lichess (4 M
# positions deja etiquetees par Stockfish 18 a 1 000 noeuds) repassee par
# Stockfish 19 a 5 000 noeuds, comme l'autojeu. Memes positions, seule
# l'etiquette change ; le resultat de la partie est garde. Arret propre avant
# la limite de 12 h (ce qui manque se reprend avec --skip).
import glob, os, shutil, subprocess, sys, threading, time, zipfile

PARTIE = 2
LIMITE = 11 * 3600 + 900          # 11 h 15
t0 = time.time()

src = glob.glob('/kaggle/input/**/label.py', recursive=True)
if not src:
    sys.exit("label.py introuvable")
root = os.path.dirname(src[0])
w = '/kaggle/working'
for f in ('label.py', 'winjob.py', 'bin_vers_csv.py', 'stockfish19'):
    shutil.copy(os.path.join(root, f), os.path.join(w, f))
if os.path.isdir(os.path.join(root, 'pylib')):
    shutil.copytree(os.path.join(root, 'pylib'), os.path.join(w, 'pylib'), dirs_exist_ok=True)
else:
    zipfile.ZipFile(os.path.join(root, 'pylib.zip')).extractall(os.path.join(w, 'pylib'))
os.chmod(os.path.join(w, 'stockfish19'), 0o755)
os.chdir(w)
STOP = os.path.join(w, 'STOP')

def minuterie():
    time.sleep(max(0, LIMITE - (time.time() - t0)))
    open(STOP, 'w').close()
threading.Thread(target=minuterie, daemon=True).start()

env = dict(os.environ, PYTHONPATH=os.path.join(w, 'pylib'))
print(subprocess.run('nproc; grep -m1 "model name" /proc/cpuinfo; echo uci | ./stockfish19 | head -1',
                     shell=True, capture_output=True, text=True).stdout, flush=True)
csv = os.path.join(w, 'partie.csv')
subprocess.run([sys.executable, 'bin_vers_csv.py', '--decode',
                os.path.join(root, f'echantillon_{PARTIE}.bin'), '--csv', csv], check=True, env=env)

nc = os.cpu_count() or 4
out = os.path.join(w, 'reetiq')
cmd = [sys.executable, '-u', 'label.py', '--csv', csv, '--engine', os.path.join(w, 'stockfish19'),
       '--out', out, '--workers', str(nc), '--nodes', '5000', '--hash', '16',
       '--min-fullmove', '0', '--quiet-only', '0', '--max-score', '100000', '--stop-file', STOP]
print(' '.join(cmd), flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
k = 0
for line in p.stdout:
    k += 1
    if k < 40 or k % 40 == 0 or 'Error' in line or 'Traceback' in line or 'termine' in line.lower():
        print(line, end='', flush=True)
p.wait()
n = sum(os.path.getsize(f) // 32 for f in glob.glob(os.path.join(out, '*.bin')))
print(f"FIN : code {p.returncode}, {n:,} positions reetiquetees, {time.time()-t0:.0f} s", flush=True)
for f in ('stockfish19', 'label.py', 'winjob.py', 'bin_vers_csv.py', 'partie.csv'):
    os.remove(os.path.join(w, f))
shutil.rmtree(os.path.join(w, 'pylib'), ignore_errors=True)
