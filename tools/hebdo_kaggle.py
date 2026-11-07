"""Entrainement continu (phase 10, D bis) : prepare les sessions Kaggle d'une semaine.

La semaine K reprend les poids du reseau precedent (meilleure epoque du point de
reprise trouve dans la sortie du noyau --depuis : final-4, ou la semaine K-1),
puis l'entraine sur des positions NEUVES avec un pas d'apprentissage reduit.
Deux sessions par semaine (quota GPU de 30 h, renouvele le samedi) : la 2e
reprend le point de reprise de la 1re. Le reseau ecrit a la fin (hebdoK.nnue)
est ensuite juge par un match au temps contre le reseau en place (relais).

    python tools/hebdo_kaggle.py --semaine 1 --depuis ovnice/onechess-final-4 \
        --pas-avant 2026-11-21T03:00 --epochs 30 [--lr 2e-4] [--sources n5-inverse]

Ecrit ci/kaggle_nuit/hebdo-K-1 et hebdo-K-2 ; rien n'est lance ici.
"""
import argparse
import json
import os

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
N5 = ('https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/'
      'nodes5000pv2_UHO.binpack')
# Jeux de positions par semaine. n5-inverse : l'autre moitie de nodes5000pv2_UHO,
# jamais vue par le final (meme graine, --inverse 1).
SOURCES = {
    'n5-inverse': [dict(url=N5, prefixe='n5i', octets=0, garde=0.5, fixe=240, graine=11, inverse=1)],
}

TETE = '''import glob as _g, os as _o, subprocess as _sp, sys as _sys, tarfile as _t, io as _io, urllib.request as _u, time as _ti, shutil as _sh
# Entrainement continu, semaine {k}, session {s} sur 2. Donnees neuves dans /tmp/donnees.
SOURCES = {sources!r}
_t0 = _ti.time()
# Journal aussi dans un fichier de sortie (conserve par Kaggle) : le journal de session a ete perdu le 7-11.
_j = open('/kaggle/working/session.log', 'a', buffering=1)
class _Tee:
    def __init__(self, *f): self.f = f
    def write(self, x): [f.write(x) for f in self.f]
    def flush(self): [f.flush() for f in self.f]
_sys.stdout = _sys.stderr = _Tee(_sys.__stdout__, _j)
_t.open(fileobj=_io.BytesIO(_u.urlopen('https://codeload.github.com/ovnice-dev/onechess/tar.gz/main').read())).extractall('/tmp/code')
TOOLS = _g.glob('/tmp/code/*/tools')[0]
for s in SOURCES:
    _sp.run([_sys.executable, '-u', TOOLS + '/hf_sources.py', '--url', s['url'], '--prefixe', s['prefixe'],
             '--dest', '/tmp/donnees', '--octets', str(s['octets']), '--garde', str(s['garde']),
             '--fixe', str(s['fixe']), '--graine', str(s['graine']), '--inverse', str(s.get('inverse', 0)),
             '--morceau', '4000000000', '--parts', str(_o.cpu_count())], check=True, stdout=_j, stderr=_sp.STDOUT)
_v = sorted(_g.glob('/tmp/donnees/*_p00_*.bin'))[0]
_o.rename(_v, '/tmp/donnees/z_val.bin')
_vr = __import__('numpy').fromfile('/tmp/donnees/z_val.bin', dtype='uint8').reshape(-1, 32)[:5_000_000]
_vr.tofile('/tmp/donnees/z_val.bin')
print(f"preparation : {{_ti.time() - _t0:.0f}} s, {{len(_g.glob('/tmp/donnees/*.bin'))}} fichiers", flush=True)
CK = 'hebdo{k}.ckpt'
_init = []
if not _g.glob('/kaggle/input/**/' + CK, recursive=True):
    # 1re session de la semaine : poids de la meilleure epoque du reseau precedent
    _prev = [p for p in _g.glob('/kaggle/input/**/*.ckpt', recursive=True)]
    if not _prev:
        _sys.exit("aucun point de reprise du reseau precedent dans /kaggle/input")
    import torch as _torch
    _c = _torch.load(_prev[0], map_location='cpu', weights_only=False)
    _torch.save(_c.get('best_state') or _c['model'], '/tmp/init.pt')
    print("poids de depart :", _prev[0], "epoque", _c.get('best', ['?'] * 3)[2], flush=True)
    _init = ['--init', '/tmp/init.pt']
EXPERIENCES = [('hebdo{k}', ['--stream', '/tmp/donnees/*.bin', '--chunk-files', '10', '--epoch-files', '{epoch_files}',
                             '--buckets', 'mat4', '--out-buckets', '4', '--out', 'hebdo{k}.nnue', '--checkpoint', CK]
                + {train!r} + _init)]
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--semaine', type=int, required=True)
    ap.add_argument('--depuis', required=True, help="noyau dont la sortie porte le point de reprise de depart")
    ap.add_argument('--pas-avant', required=True, help="samedi du quota neuf, UTC (ex. 2026-11-21T03:00)")
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', default='2e-4')
    ap.add_argument('--hidden', default='1024')
    ap.add_argument('--epoch-files', type=int, default=60)
    ap.add_argument('--sources', default='n5-inverse', choices=list(SOURCES))
    ap.add_argument('--limite', type=int, default=10 * 3600)
    ap.add_argument('--dossier', default=os.path.join(RACINE, 'ci', 'kaggle_nuit'))
    a = ap.parse_args()

    corps = open(os.path.join(RACINE, '_kaggle', 'onechess-1-18-rn5', 'run.py'), encoding='utf-8').read()
    corps = corps[corps.index('\n\n# Entrainement Onechess'):]
    corps = corps.replace('LIMITE = 11 * 3600 + 15 * 60', f'LIMITE = {a.limite}')
    corps = corps.replace("""bins = [b for b in glob.glob('/kaggle/input/**/*.bin', recursive=True)]
if not bins:
    sys.exit("donnees introuvables dans /kaggle/input")
root = os.path.dirname(bins[0])""", "root = '/tmp/donnees'")
    assert "root = '/tmp/donnees'" in corps
    train = ['--hidden', a.hidden, '--epochs', str(a.epochs), '--lr', a.lr]
    k = a.semaine
    for s in (1, 2):
        nom = f'hebdo-{k}-{s}'
        d = os.path.join(a.dossier, nom)
        os.makedirs(d, exist_ok=True)
        tete = TETE.format(k=k, s=s, sources=SOURCES[a.sources], epoch_files=a.epoch_files, train=train)
        with open(os.path.join(d, 'run.py'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(tete + corps)
        prec = a.depuis if s == 1 else f'ovnice/onechess-hebdo-{k}-1'
        meta = {"id": f"ovnice/onechess-{nom}", "title": f"onechess-{nom}", "code_file": "run.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                "enable_internet": True, "dataset_sources": ["ovnice/onechess-train"],
                "machine_shape": "NvidiaTeslaT4", "competition_sources": [], "kernel_sources": [prec]}
        with open(os.path.join(d, 'kernel-metadata.json'), 'w', newline='\n') as f:
            json.dump(meta, f, indent=1)
        with open(os.path.join(d, 'apres'), 'w', newline='\n') as f:
            f.write(prec + '\n')
        with open(os.path.join(d, 'pas_avant'), 'w', newline='\n') as f:
            f.write(a.pas_avant + '\n')
        print('ecrit', d)


if __name__ == '__main__':
    main()
