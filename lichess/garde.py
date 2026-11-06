"""Garde-fou d'OnechessBot (toutes les 5 minutes).

Le 6-11, une partie perdue au temps a laisse lichess-bot croire qu'elle durait
encore : plus aucune partie pendant 10 h. Ici : si lichess-bot occupe sa place
de partie (journal « Process Freed. Count: 1 » ou « Process Used ») alors que
Lichess ne voit aucune partie en cours, deux fois de suite (10 minutes), on
redemarre lichess-bot. Journal ~/garde.log. Le jeton n'est jamais affiche.
"""
import json
import os
import re
import subprocess
import time
import urllib.request

ICI = os.path.expanduser("~")
CFG = os.path.join(ICI, "lichess-bot", "config.yml")
ETAT = os.path.join(ICI, "garde_etat.json")
JOURNAL = os.path.join(ICI, "garde.log")


def note(msg):
    with open(JOURNAL, "a") as f:
        f.write(time.strftime("%d-%m %H:%M ") + msg + "\n")


def en_cours_lichess():
    jeton = re.search(r"^token:\s*\"?([^\"\n]+)\"?", open(CFG).read(), re.M).group(1)
    req = urllib.request.Request("https://lichess.org/api/account/playing",
                                 headers={"Authorization": "Bearer " + jeton, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return len(json.loads(r.read().decode()).get("nowPlaying", []))


def occupe_bot():
    """Nombre de parties que lichess-bot croit avoir, d'apres sa derniere ligne de comptage."""
    log = subprocess.run(["journalctl", "-u", "lichessbot", "--since", "-30min", "--no-pager", "-o", "cat"],
                         capture_output=True, text=True).stdout
    n = re.findall(r"Process (?:Freed|Used|Queued)\. Count: (\d+)", log)
    return int(n[-1]) if n else 0


def main():
    e = json.load(open(ETAT)) if os.path.exists(ETAT) else {"suspect": 0}
    try:
        lichess = en_cours_lichess()
    except Exception as x:
        note(f"API Lichess injoignable : {x}")
        return
    bot = occupe_bot()
    if bot > 0 and lichess == 0:
        e["suspect"] += 1
        if e["suspect"] >= 2:
            note(f"lichess-bot croit jouer {bot} partie(s), Lichess n'en voit aucune depuis 10 min : redemarrage")
            subprocess.run(["sudo", "systemctl", "restart", "lichessbot"], check=False)
            e["suspect"] = 0
    else:
        e["suspect"] = 0
    json.dump(e, open(ETAT, "w"))


if __name__ == "__main__":
    main()
