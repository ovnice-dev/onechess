# Autojeu Stockfish 19 contre Stockfish 19 sur un processeur Kaggle
# (tools/datagen.py) : 5000 noeuds par coup, chaque position calme est
# etiquetee par le score de la recherche qui choisit le coup, puis par le
# resultat de la partie. Arret propre avant la limite de 12 h.
import glob, os, shutil, subprocess, sys, threading, time

GRAINE = 74000                     # une graine differente par noyau
LIMITE = 11 * 3600 + 900          # 11 h 15
t0 = time.time()

src = glob.glob('/kaggle/input/**/datagen.py', recursive=True)
if not src:
    sys.exit("datagen.py introuvable")
root = os.path.dirname(src[0])
w = '/kaggle/working'
for f in ('datagen.py', 'winjob.py', 'stockfish19', 'openings.epd'):
    shutil.copy(os.path.join(root, f), os.path.join(w, f))
shutil.copytree(os.path.join(root, 'pylib'), os.path.join(w, 'pylib'), dirs_exist_ok=True)
os.chmod(os.path.join(w, 'stockfish19'), 0o755)
os.chdir(w)
STOP = os.path.join(w, 'STOP')

def minuterie():
    time.sleep(max(0, LIMITE - (time.time() - t0)))
    open(STOP, 'w').close()
threading.Thread(target=minuterie, daemon=True).start()

print(subprocess.run('nproc; grep -m1 "model name" /proc/cpuinfo; free -g | head -2; '
                     'echo uci | ./stockfish19 | head -1', shell=True,
                     capture_output=True, text=True).stdout, flush=True)
nc = os.cpu_count() or 4
env = dict(os.environ, PYTHONPATH=os.path.join(w, 'pylib'))
out = os.path.join(w, 'auto')
os.makedirs(out, exist_ok=True)
cmd = [sys.executable, '-u', 'datagen.py', '--engine', os.path.join(w, 'stockfish19'),
       '--openings', os.path.join(w, 'openings.epd'), '--out', out,
       '--positions', '1000000000', '--workers', str(nc), '--nodes', '5000',
       '--hash', '16', '--seed', str(GRAINE), '--stop-file', STOP]
print(' '.join(cmd), flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
k = 0
for line in p.stdout:
    k += 1
    if k < 40 or k % 60 == 0 or 'termine' in line or 'Error' in line or 'Traceback' in line:
        print(line, end='', flush=True)
p.wait()
n = sum(os.path.getsize(f) // 32 for f in glob.glob(os.path.join(out, '*.bin')))
print(f"FIN : code {p.returncode}, {n:,} positions, {time.time()-t0:.0f} s", flush=True)
for f in ('stockfish19', 'datagen.py', 'winjob.py', 'openings.epd'):
    os.remove(os.path.join(w, f))
shutil.rmtree(os.path.join(w, 'pylib'), ignore_errors=True)
