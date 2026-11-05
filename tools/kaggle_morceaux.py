"""Morceaux de SPRT joues sur les sessions processeur de Kaggle.

Chaque noyau (onechess-m-<test>-<k>, processeur) recompile le moteur du jeu
ovnice/onechess-code, verifie la signature, puis joue des parties du test au
temps, sans arret anticipe, jusqu'a la limite de session. Son fichier .state,
au meme format que ceux de GitHub, se range ensuite dans
ci_resultats/lot/<test>/kaggle/ (ci/recolter_kaggle_lot.sh) : le bilan du lot
l'additionne aux vagues GitHub.

    python tools/kaggle_morceaux.py --test K1ExtT "K1Ext=1" --test R6MarginT "R6Margin=100" \
        [--apres ovnice/onechess-autojeu-1 ...]

Les dossiers sont ecrits dans ci/kaggle_nuit/m-<test>-<k> ; le relais les lance.
"""
import argparse
import json
import os

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUN = r'''# Morceau de SPRT sur processeur Kaggle : {nom}, graine {graine}.
import glob, os, shutil, signal, subprocess, sys, time
NOM, OPT1, OPT2, MODE, GRAINE = {nom!r}, {opt1!r}, {opt2!r}, {mode!r}, {graine}
SIGNATURE = {signature}
LIMITE = 11 * 3600            # arret propre avant les 12 h de la session
T0 = time.time()

src = glob.glob('/kaggle/input/**/arena.py', recursive=True)
if not src:
    sys.exit("code introuvable dans /kaggle/input")
w = '/kaggle/working/onechess'
shutil.copytree(os.path.dirname(src[0]), w, dirs_exist_ok=True)
os.chdir(w)
pext = '-DUSE_PEXT' if ' bmi2 ' in open('/proc/cpuinfo').read() else ''
r = subprocess.run(f'g++ -std=c++20 -O3 -DNDEBUG -flto=auto -fno-exceptions -march=native {{pext}} '
                   f'-pthread -I. *.cpp tbprobe.c -o onechess', shell=True)
if r.returncode:
    sys.exit("echec de compilation")
out = subprocess.run("ulimit -s 65536; printf 'setoption name EvalFile value onechess.nnue\\nbench 12\\nquit\\n' "
                     "| ./onechess", shell=True, text=True, capture_output=True).stdout
n = [l for l in out.splitlines() if 'Noeuds totaux' in l]
n = int(n[0].split(':')[1]) if n else -1
print(f"signature Linux {{n}}, attendue {{SIGNATURE}}", flush=True)
if n != SIGNATURE:
    sys.exit("SIGNATURE DIFFERENTE")

nc = max(1, (os.cpu_count() or 4) // 2)      # au temps : une partie par coeur physique
o1 = ' '.join(f'--opt1 {{o}}' for o in OPT1.split())
o2 = ' '.join(f'--opt2 {{o}}' for o in OPT2.split())
cmd = (f'ulimit -s 65536; PYTHONPATH={{w}}/pychess.whl:{{w}}/pylib exec python -u arena.py --e1 ./onechess --e2 ./onechess '
       f'--name1 {{NOM}} --name2 Reference --opt1 EvalFile=onechess.nnue --opt2 EvalFile=onechess.nnue '
       f'{{o1}} {{o2}} --tc {{MODE[3:]}} --hash 64 --games 100000 --concurrency {{nc}} --seed {{GRAINE}} '
       f'--quiet --openings uho.epd --state /kaggle/working/{{NOM}}-kaggle-{{GRAINE}}.state')
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
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', nargs=2, action='append', metavar=('NOM', 'OPTS'), required=True)
    ap.add_argument('--mode', default='tc:10+0.1')
    ap.add_argument('--apres', nargs='*', default=[])
    ap.add_argument('--graine', type=int, default=900000)
    ap.add_argument('--suffixe', default='', help='ex. -2 : nouveaux noyaux (un noyau Kaggle ne se relance pas)')
    a = ap.parse_args()
    signature = int(open(os.path.join(RACINE, 'ci', 'signature.txt')).read().strip())
    for k, (nom, opts) in enumerate(a.test):
        d = os.path.join(RACINE, 'ci', 'kaggle_nuit', f'm-{nom.lower()}{a.suffixe}')
        os.makedirs(d, exist_ok=True)
        graine = a.graine + k
        with open(os.path.join(d, 'run.py'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(RUN.format(nom=nom, opt1=opts, opt2='', mode=a.mode, graine=graine, signature=signature))
        meta = {"id": f"ovnice/onechess-m-{nom.lower()}{a.suffixe}", "title": f"onechess-m-{nom.lower()}{a.suffixe}",
                "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": False, "enable_internet": False, "dataset_sources": ["ovnice/onechess-code"],
                "competition_sources": [], "kernel_sources": []}
        with open(os.path.join(d, 'kernel-metadata.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        open(os.path.join(d, 'cpu'), 'w').close()
        if k < len(a.apres):
            with open(os.path.join(d, 'apres'), 'w') as f:
                f.write(a.apres[k] + '\n')
        print('ecrit', d, 'graine', graine)


if __name__ == '__main__':
    main()
