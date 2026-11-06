# Morceau de SPRT sur processeur Kaggle : FutCorrT, graine 910003.
import glob, os, shutil, signal, subprocess, sys, time
NOM, OPT1, OPT2, MODE, GRAINE = 'FutCorrT', 'FutCorr=32', '', 'tc:10+0.1', 910003
SIGNATURE = 1130532
LIMITE = 11 * 3600            # arret propre avant les 12 h de la session
T0 = time.time()

src = glob.glob('/kaggle/input/**/arena.py', recursive=True)
if not src:
    sys.exit("code introuvable dans /kaggle/input")
w = '/kaggle/working/onechess'
shutil.copytree(os.path.dirname(src[0]), w, dirs_exist_ok=True)
os.chdir(w)
pext = '-DUSE_PEXT' if ' bmi2 ' in open('/proc/cpuinfo').read() else ''
r = subprocess.run(f'g++ -std=c++20 -O3 -DNDEBUG -flto=auto -fno-exceptions -march=native {pext} '
                   f'-pthread -I. *.cpp tbprobe.c -o onechess', shell=True)
if r.returncode:
    sys.exit("echec de compilation")
out = subprocess.run("ulimit -s 65536; printf 'setoption name EvalFile value onechess.nnue\\nbench 12\\nquit\\n' "
                     "| ./onechess", shell=True, text=True, capture_output=True).stdout
n = [l for l in out.splitlines() if 'Noeuds totaux' in l]
n = int(n[0].split(':')[1]) if n else -1
print(f"signature Linux {n}, attendue {SIGNATURE}", flush=True)
if n != SIGNATURE:
    sys.exit("SIGNATURE DIFFERENTE")

nc = max(1, (os.cpu_count() or 4) // 2)      # au temps : une partie par coeur physique
o1 = ' '.join(f'--opt1 {o}' for o in OPT1.split())
o2 = ' '.join(f'--opt2 {o}' for o in OPT2.split())
cmd = (f'ulimit -s 65536; PYTHONPATH={w}/pychess.whl:{w}/pylib exec python -u arena.py --e1 ./onechess --e2 ./onechess '
       f'--name1 {NOM} --name2 Reference --opt1 EvalFile=onechess.nnue --opt2 EvalFile=onechess.nnue '
       f'{o1} {o2} --tc {MODE[3:]} --hash 64 --games 100000 --concurrency {nc} --seed {GRAINE} '
       f'--quiet --openings uho.epd --state /kaggle/working/{NOM}-kaggle-{GRAINE}.state')
print('$', cmd, flush=True)
p = subprocess.Popen(cmd, shell=True)
while p.poll() is None:
    time.sleep(30)
    if time.time() - T0 > LIMITE:
        p.send_signal(signal.SIGINT)      # l'arene garde son etat, paire par paire
        try:
            p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            p.kill()
        break
print("fin", time.time() - T0, flush=True)
