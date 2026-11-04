import glob as _g, os as _o, subprocess as _sp, sys as _sys, tarfile as _t, io as _io, urllib.request as _u, time as _ti
# Reseau final, session 4 sur 4. Donnees publiques converties dans /tmp/donnees
# (outils du depot public) ; un fichier mis de cote pour la validation.
SOURCES = [{'url': 'https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/nodes5000pv2_UHO.binpack', 'prefixe': 'n5', 'octets': 0, 'garde': 0.5, 'fixe': 240, 'graine': 11}]
_t0 = _ti.time()
_t.open(fileobj=_io.BytesIO(_u.urlopen('https://codeload.github.com/ovnice-dev/onechess/tar.gz/main').read())).extractall('/tmp/code')
TOOLS = _g.glob('/tmp/code/*/tools')[0]
for s in SOURCES:
    _sp.run([_sys.executable, '-u', TOOLS + '/hf_sources.py', '--url', s['url'], '--prefixe', s['prefixe'],
             '--dest', '/tmp/donnees', '--octets', str(s['octets']), '--garde', str(s['garde']),
             '--fixe', str(s['fixe']), '--graine', str(s['graine']), '--parts', str(_o.cpu_count())], check=True)
_v = sorted(_g.glob('/tmp/donnees/*_p00_*.bin'))[0]
_o.rename(_v, '/tmp/donnees/z_val.bin')
_vr = __import__('numpy').fromfile('/tmp/donnees/z_val.bin', dtype='uint8').reshape(-1, 32)[:5_000_000]
_vr.tofile('/tmp/donnees/z_val.bin')
print(f"preparation : {_ti.time() - _t0:.0f} s, {len(_g.glob('/tmp/donnees/*.bin'))} fichiers", flush=True)
EXPERIENCES = [('final', ['--stream', '/tmp/donnees/*.bin', '--chunk-files', '10', '--epoch-files', '60',
                          '--buckets', 'mat4', '--out-buckets', '4', '--out', 'final.nnue', '--checkpoint', 'final.ckpt']
                + ['--hidden', '1024', '--epochs', '60'])]


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

LIMITE = 36000     # 11 h 15 : il reste de quoi ecrire la sortie
T0 = time.time()                 # la preparation du lot compte dans la session

root = '/tmp/donnees'
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
