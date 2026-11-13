"""Prepare la chaine de sessions Kaggle du reseau final pour le relais horaire.

Chaque session (noyau onechess-final-k) : telecharge et convertit les donnees
publiques dans /tmp/donnees (tools/hf_sources.py, deterministe), met un fichier
de cote pour la validation, reprend le point de reprise de la session k-1
(kernel_sources) et entraine jusqu'a la limite de session. Le relais
(.github/workflows/kaggle_nuit.yml) lance la session k quand la k-1 est finie.

    python tools/final_kaggle.py --sessions 4 --train "--hidden 1024 --epochs 60" \
        [--apres ovnice/onechess-1-24-r1024h]

Les dossiers sont ecrits dans ci/kaggle_nuit/final-k ; rien n'est lance ici.
"""
import argparse
import json
import os

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)

TETE = '''import glob as _g, os as _o, subprocess as _sp, sys as _sys, tarfile as _t, io as _io, urllib.request as _u, time as _ti
# Reseau final, session {k} sur {n}. Donnees publiques converties dans /tmp/donnees
# (outils du depot public) ; un fichier mis de cote pour la validation.
SOURCES = {sources!r}
_t0 = _ti.time()
# Journal aussi dans un fichier de sortie (conserve par Kaggle) : le journal de session a ete perdu le 7-11.
_j = open('/kaggle/working/session.log', 'a', buffering=1)
class _Tee:
    def __init__(self, *f): self.f = f
    def write(self, x): [f.write(x) for f in self.f]
    def flush(self): [f.flush() for f in self.f]
_sys.stdout = _sys.stderr = _Tee(_sys.__stdout__, _j)
_t.open(fileobj=_io.BytesIO(_u.urlopen('https://codeload.github.com/ovnice-dev/onechess-tools/tar.gz/main').read())).extractall('/tmp/code')
TOOLS = _g.glob('/tmp/code/*/tools')[0]
for s in SOURCES:
    _sp.run([_sys.executable, '-u', TOOLS + '/hf_sources.py', '--url', s['url'], '--prefixe', s['prefixe'],
             '--dest', '/tmp/donnees', '--octets', str(s['octets']), '--garde', str(s['garde']),
             '--fixe', str(s['fixe']), '--graine', str(s['graine']), '--morceau', '4000000000', '--parts', str(_o.cpu_count())], check=True, stdout=_j, stderr=_sp.STDOUT)
_v = sorted(_g.glob('/tmp/donnees/*_p00_*.bin'))[0]
_o.rename(_v, '/tmp/donnees/z_val.bin')
_vr = __import__('numpy').fromfile('/tmp/donnees/z_val.bin', dtype='uint8').reshape(-1, 32)[:5_000_000]
_vr.tofile('/tmp/donnees/z_val.bin')
print(f"preparation : {{_ti.time() - _t0:.0f}} s, {{len(_g.glob('/tmp/donnees/*.bin'))}} fichiers", flush=True)
EXPERIENCES = [('final', ['--stream', '/tmp/donnees/*.bin', '--chunk-files', '10', '--epoch-files', '{epoch_files}',
                          '--buckets', 'mat4', '--out-buckets', '4', '--out', 'final.nnue', '--checkpoint', 'final.ckpt']
                + {train!r})]
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', type=int, default=4)
    ap.add_argument('--train', default='--hidden 1024 --epochs 60')
    ap.add_argument('--epoch-files', type=int, default=60)
    ap.add_argument('--apres', default='')
    # quota GPU : 30 h par semaine (renouvele le samedi d'apres nos releves), soit 2 sessions
    # de ~10 h 30 ; les sessions 1-2 partent apres --pas-avant, les 3-4 une semaine plus tard, etc.
    ap.add_argument('--pas-avant', default='2026-11-07T03:00')
    ap.add_argument('--par-semaine', type=int, default=2)
    ap.add_argument('--limite', type=int, default=10 * 3600)
    ap.add_argument('--suffixe', default='', help="ex. b : nouveaux noyaux onechess-finalb-k (un noyau en erreur ne se relance pas)")   # + ~30 min de preparation, + 1 epoque au plus
    a = ap.parse_args()

    sources = [dict(url='https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/'
                        'nodes5000pv2_UHO.binpack',
                    prefixe='n5', octets=7_000_000_000, garde=1.0, fixe=240, graine=11)]
    # 7-11 : le fichier entier donne ~6 G positions a garde 0,5 (~195 Go) ; une session Kaggle n'a que ~100 Go
    # de disque. Les 7 premiers Go : ~2,1 G positions (~68 Go) ; le reste est la reserve de l'entrainement continu.
    corps = open(os.path.join(RACINE, '_kaggle', 'onechess-1-18-rn5', 'run.py'), encoding='utf-8').read()
    corps = corps[corps.index('\n\n# Entrainement Onechess'):]
    corps = corps.replace('LIMITE = 11 * 3600 + 15 * 60', f'LIMITE = {a.limite}')
    # les donnees sont dans /tmp/donnees, pas dans /kaggle/input
    corps = corps.replace("""bins = [b for b in glob.glob('/kaggle/input/**/*.bin', recursive=True)]
if not bins:
    sys.exit("donnees introuvables dans /kaggle/input")
root = os.path.dirname(bins[0])""", "root = '/tmp/donnees'")
    assert "root = '/tmp/donnees'" in corps
    for k in range(1, a.sessions + 1):
        d = os.path.join(RACINE, 'ci', 'kaggle_nuit', f'final{a.suffixe}-{k}')
        os.makedirs(d, exist_ok=True)
        tete = TETE.format(k=k, n=a.sessions, sources=sources, epoch_files=a.epoch_files,
                           train=a.train.split())
        with open(os.path.join(d, 'run.py'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(tete + corps)
        meta = {"id": f"ovnice/onechess-final{a.suffixe}-{k}", "title": f"onechess-final{a.suffixe}-{k}", "code_file": "run.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                "enable_internet": True, "dataset_sources": ["ovnice/onechess-train"],
                "machine_shape": "NvidiaTeslaT4", "competition_sources": [],
                "kernel_sources": [f"ovnice/onechess-final{a.suffixe}-{k - 1}"] if k > 1 else []}
        with open(os.path.join(d, 'kernel-metadata.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        apres = f"ovnice/onechess-final{a.suffixe}-{k - 1}" if k > 1 else a.apres
        if apres:
            with open(os.path.join(d, 'apres'), 'w') as f:
                f.write(apres + '\n')
        if a.pas_avant:
            import datetime
            t = datetime.datetime.fromisoformat(a.pas_avant) + datetime.timedelta(weeks=(k - 1) // a.par_semaine)
            with open(os.path.join(d, 'pas_avant'), 'w') as f:
                f.write(t.strftime('%Y-%m-%dT%H:%M') + '\n')
        print('ecrit', d)


if __name__ == '__main__':
    main()
