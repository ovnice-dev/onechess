"""Mise a jour automatique d'OnechessBot depuis les releases GitHub (toutes les heures).

Une release du depot public ovnice-dev/onechess porte :
  - son etiquette (tag), posee sur le code a compiler ;
  - onechess.nnue (le reseau) et signature.txt (bench 12 attendu).
Quand une release plus recente que la derniere installee apparait :
  1. compile le moteur ARM a cette etiquette (~/onechess), telecharge le reseau ;
  2. verifie la signature bench 12 ; sinon, rien n'est installe ;
  3. attend qu'aucune partie ne soit en cours, installe (remplacement atomique,
     anciens fichiers gardes en .precedent) et redemarre lichess-bot : il garde
     son moteur d'une partie a l'autre, le redemarrage est indispensable.
Journal ~/maj_bot.log, etat ~/maj_bot_etat.json. --essai : dit ce qu'il ferait.
Le jeton Lichess (lu dans la configuration de lichess-bot) n'est jamais affiche.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

DEPOT = "ovnice-dev/onechess"
API = os.environ.get("ONE_MAJ_API", f"https://api.github.com/repos/{DEPOT}/releases/latest")   # essais
ICI = os.path.expanduser("~")
SRC = os.path.join(ICI, "onechess")
MOTEURS = os.path.join(ICI, "lichess-bot", "engines")
CFG = os.path.join(ICI, "lichess-bot", "config.yml")
ETAT = os.path.join(ICI, "maj_bot_etat.json")
JOURNAL = os.path.join(ICI, "maj_bot.log")
ESSAI = "--essai" in sys.argv


def note(msg):
    ligne = time.strftime("%d-%m %H:%M ") + msg
    print(ligne, flush=True)
    if not ESSAI:
        with open(JOURNAL, "a") as f:
            f.write(ligne + "\n")


def lire(url, accept="application/vnd.github+json", jeton=None):
    h = {"Accept": accept, "User-Agent": "onechess-maj-bot"}
    if jeton:
        h["Authorization"] = "Bearer " + jeton
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
        return r.read()


def partie_en_cours():
    jeton = re.search(r"^token:\s*\"?([^\"\n]+)\"?", open(CFG).read(), re.M).group(1)
    d = json.loads(lire("https://lichess.org/api/account/playing", "application/json", jeton))
    return bool(d.get("nowPlaying"))


def sh(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)


def main():
    etat = json.load(open(ETAT)) if os.path.exists(ETAT) else {}
    try:
        rel = json.loads(lire(API))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("aucune release publiee")
            return
        raise
    tag = rel["tag_name"]
    if tag == etat.get("installe"):
        return
    if tag in etat.get("refusees", []):
        return
    assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
    if "onechess.nnue" not in assets or "signature.txt" not in assets:
        note(f"release {tag} incomplete (onechess.nnue ou signature.txt absent) : ignoree")
        return
    note(f"nouvelle release {tag}")
    if ESSAI:
        print("essai : compilerait", tag, "et installerait", list(assets))
        return

    # 1. Code a l'etiquette, compilation ARM, reseau
    r = sh(f"git fetch -q --tags origin && git checkout -q {tag}", SRC)
    if r.returncode:
        note(f"git : {r.stderr.strip()[:200]}")
        return
    neuf = os.path.join(SRC, "engine", "onechess.neuf")
    r = sh("g++ -std=c++20 -O3 -DNDEBUG -flto=auto -fno-exceptions -march=native -pthread "
           f"-Ifathom src/*.cpp fathom/tbprobe.c -o {neuf}", os.path.join(SRC, "engine"))
    if r.returncode:
        note(f"compilation echouee : {r.stderr.strip()[-300:]}")
        return
    reseau = os.path.join(MOTEURS, "onechess.nnue.neuf")
    with open(reseau, "wb") as f:
        f.write(lire(assets["onechess.nnue"], "application/octet-stream"))
    attendu = int(lire(assets["signature.txt"], "application/octet-stream").decode().strip())

    # 2. Signature
    out = sh(f"printf 'setoption name EvalFile value {reseau}\\nbench 12\\nquit\\n' | {neuf}").stdout
    m = re.search(r"Noeuds totaux\s*:\s*(\d+)", out)
    n = int(m.group(1)) if m else -1
    if n != attendu:
        note(f"{tag} : signature {n}, attendue {attendu} : RIEN INSTALLE")
        etat.setdefault("refusees", []).append(tag)
        json.dump(etat, open(ETAT, "w"))
        return

    # 3. Installation quand le bot est libre (au plus 3 h d'attente, sinon a l'heure suivante)
    t0 = time.time()
    while partie_en_cours():
        if time.time() - t0 > 3 * 3600:
            note(f"{tag} : partie toujours en cours apres 3 h, nouvel essai plus tard")
            return
        time.sleep(30)
    for nom, src in (("onechess", neuf), ("onechess.nnue", reseau)):
        dst = os.path.join(MOTEURS, nom)
        if os.path.exists(dst):
            shutil.copy2(dst, dst + ".precedent")
        tmp = dst + ".tmp"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    subprocess.run(["sudo", "systemctl", "restart", "lichessbot"], check=False)
    etat["installe"] = tag
    json.dump(etat, open(ETAT, "w"))
    note(f"{tag} installe (signature {n}), lichess-bot redemarre")


if __name__ == "__main__":
    main()
