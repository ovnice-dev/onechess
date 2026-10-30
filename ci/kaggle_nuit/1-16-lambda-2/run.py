PREP = ['--data', '/kaggle/input/**/onechess-data/**/*.bin', '--data', '/kaggle/input/**/onechess-autojeu-data/**/*.bin', '--out', '/tmp/lot', '--files', '128', '--dedup', '4', '--seed', '0', '--marque', '/kaggle/input/**/onechess-autojeu-data/**/*.bin']
EXPERIENCES = [('lambda', ['--stream', '/tmp/lot/*.bin', '--chunk-files', '16', '--epochs', '20', '--buckets', 'mat4', '--lambda-auto', '0.6', '--out', 'lambda.nnue', '--save-pt', 'lambda.pt', '--checkpoint', 'lambda.ckpt'])]


# Entrainement Onechess sur GPU Kaggle, en plusieurs sessions si besoin.
#
# Une session Kaggle est tuee au bout de 12 h. train.py ecrit un point de
# reprise a chaque epoque (--checkpoint) : on arrete donc l'entrainement
# proprement avant la limite (LIMITE), on rend la main, et la session suivante
# -- un noyau qui attache la sortie de celui-ci (kernel_sources) -- retrouve le
# point de reprise dans /kaggle/input et repart a l'epoque suivante.
#
# EXPERIENCES est defini en tete par le fichier copie (une seule experience
# conseillee par noyau long).
import glob, os, shutil, signal, subprocess, sys, time

LIMITE = 11 * 3600 + 15 * 60     # 11 h 15 : il reste de quoi ecrire la sortie
T0 = time.time()                 # la preparation du lot compte dans la session

bins = [b for b in glob.glob('/kaggle/input/**/*.bin', recursive=True)]
if not bins:
    sys.exit("donnees introuvables dans /kaggle/input")
root = os.path.dirname(bins[0])
code = glob.glob('/kaggle/input/**/TRAIN_CODE', recursive=True)
croot = os.path.dirname(code[0]) if code else root
print("code d'entrainement :", croot, flush=True)
os.chdir('/kaggle/working')
for f in ('train.py', 'features.py', 'menaces.py', 'prep_lot.py'):
    if os.path.exists(os.path.join(croot, f)):
        shutil.copy(os.path.join(croot, f), f)
data = os.path.join(root, '*.bin')
print("donnees :", len(glob.glob(data)), "fichiers dans", root, flush=True)
subprocess.run(['nvidia-smi'])

# PREP (facultatif, defini en tete comme EXPERIENCES) : preparation du lot pour
# train.py --stream. Deterministe (graine fixe) : refaite a chaque session, elle
# redonne exactement les memes fichiers, donc la reprise reste exacte.
PREP = globals().get('PREP')
if PREP:
    print("disque /tmp :", shutil.disk_usage('/tmp'), flush=True)
    print("disque /kaggle/working :", shutil.disk_usage('/kaggle/working'), flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, '-u', 'prep_lot.py'] + PREP)
    if r.returncode:
        sys.exit(f"preparation du lot en echec (code {r.returncode})")
    print(f"preparation du lot : {time.time() - t:.0f} s", flush=True)

t_debut = T0
for nom, args in EXPERIENCES:
    # Point de reprise d'une session precedente ?
    ck = args[args.index('--checkpoint') + 1] if '--checkpoint' in args else None
    if ck:
        prev = [p for p in glob.glob('/kaggle/input/**/' + ck, recursive=True)]
        if prev and not os.path.exists(ck):
            shutil.copy(prev[0], ck)
            print(f"point de reprise trouve : {prev[0]}", flush=True)
    cmd = [sys.executable, '-u', 'train.py', '--data', data, '--out-buckets', '4'] + args
    print(f"\n===== {nom} : {' '.join(args)}", flush=True)
    reste = LIMITE - (time.time() - t_debut)
    with open(f'{nom}.log', 'a') as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end='', flush=True)
            log.write(line)
            if time.time() - t_debut > LIMITE:
                print("===== LIMITE DE SESSION : arret, a reprendre dans la session suivante", flush=True)
                p.send_signal(signal.SIGTERM)
                break
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
    print(f"===== {nom} : code {p.returncode}, {time.time() - t_debut:.0f} s", flush=True)
