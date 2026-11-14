# SPRT a noeuds fixes d'Onechess sur un processeur Kaggle (Linux).
# Le code vient du jeu de donnees prive ovnice/onechess-code. Le moteur est
# recompile ici ; sa signature bench doit etre identique a celle de Windows,
# sinon le test s'arrete : un moteur qui ne cherche pas pareil ne teste rien.
# TEST (nom, options e1, options e2) est defini en tete par le generateur.
import glob, os, shutil, subprocess, sys

SIGNATURE = 1006118

src = glob.glob('/kaggle/input/**/arena.py', recursive=True)
if not src:
    sys.exit("code introuvable dans /kaggle/input")
root = os.path.dirname(src[0])
w = '/kaggle/working/onechess'
shutil.copytree(root, w, dirs_exist_ok=True)
os.chdir(w)

def sh(cmd):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout, flush=True)
    return r

cpu = open('/proc/cpuinfo').read()
pext = '-DUSE_PEXT' if ' bmi2 ' in cpu else ''
sh('nproc; grep -m1 "model name" /proc/cpuinfo; g++ --version | head -1')
r = sh(f'g++ -std=c++20 -O3 -DNDEBUG -flto=auto -fno-exceptions -march=native {pext} '
       f'-pthread -I. *.cpp tbprobe.c -o onechess')
if r.returncode:
    sys.exit("echec de compilation")

r = sh("ulimit -s 65536; printf 'setoption name EvalFile value onechess.nnue\\nbench 12\\nquit\\n' | ./onechess | tail -3")
noeuds = [l for l in r.stdout.splitlines() if 'Noeuds totaux' in l]
n = int(noeuds[0].split(':')[1]) if noeuds else -1
print(f"signature Linux {n}, Windows {SIGNATURE}", flush=True)
if n != SIGNATURE:
    sys.exit("SIGNATURE DIFFERENTE : le moteur Linux ne cherche pas comme celui de Windows")

OUVERTURES = 'uho.epd' if os.path.exists('uho.epd') else 'openings.epd'
nom, o1, o2 = TEST[:3]
bornes = TEST[4] if len(TEST) > 4 else "0 8"
mode = TEST[3] if len(TEST) > 3 else 'nodes'
opts = ' '.join(f'--opt1 {o}' for o in o1) + ' ' + ' '.join(f'--opt2 {o}' for o in o2)
nc = os.cpu_count() or 4
if mode.startswith('tc:'):
    # Au temps : une partie par coeur physique, sinon les deux moteurs d'une
    # meme machine se volent du temps et le test ne mesure plus rien.
    cadence = f'--tc {mode[3:]} --hash 64'
    nc = max(1, nc // 2)
else:
    cadence = '--nodes1 100000 --nodes2 100000 --hash 32'
cmd = (f'ulimit -s 65536; PYTHONPATH={w}/pylib python -u arena.py --e1 ./onechess --e2 ./onechess '
       f'--name1 {nom} --name2 Reference --opt1 EvalFile=onechess.nnue --opt2 EvalFile=onechess.nnue '
       f'{opts} {cadence} --games 20000 --concurrency {nc} '
       f'--openings {OUVERTURES} --sprt {bornes} --state /kaggle/working/{nom}.state')
print('$', cmd, flush=True)
with open(f'/kaggle/working/{nom}.log', 'w') as log:
    p = subprocess.Popen(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        log.write(line)
        log.flush()
        mots = line.split()
        # une ligne d'avancement sur 100 parties ; tout le reste s'affiche
        if len(mots) > 1 and mots[1] == 'parties' and mots[0].isdigit() and int(mots[0]) % 100:
            continue
        print(line, end='', flush=True)
    p.wait()
