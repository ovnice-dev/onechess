"""Cadences des defis d'OnechessBot (toutes les minutes).

lichess-bot tire la cadence de ses defis au hasard dans une liste. Ici, on
passe a la categorie suivante (bullet -> blitz -> rapide -> bullet...) quand
10 defis de suite n'ont donne aucune partie ; une partie jouee garde la
categorie. Le changement redemarre lichess-bot (2 minutes avant le defi suivant).
"""
import json
import os
import re
import subprocess
import time

ICI = os.path.expanduser("~")
CFG = os.path.join(ICI, "lichess-bot", "config.yml")
ETAT = os.path.join(ICI, "categories_etat.json")
JOURNAL = os.path.join(ICI, "categories.log")
ESSAIS = 10
CATEGORIES = [("bullet", [60, 120], [1, 2]),
              ("blitz", [180, 300], [2, 3]),
              ("rapide", [600, 900], [5, 10])]


def note(msg):
    with open(JOURNAL, "a") as f:
        f.write(time.strftime("%d-%m %H:%M ") + msg + "\n")


def appliquer(i):
    _, bases, incs = CATEGORIES[i]
    s = open(CFG).read()
    s = re.sub(r"  challenge_initial_time:\n(    - \d+\n)+",
               "  challenge_initial_time:\n" + "".join(f"    - {b}\n" for b in bases), s)
    s = re.sub(r"  challenge_increment:\n(    - \d+\n)+",
               "  challenge_increment:\n" + "".join(f"    - {c}\n" for c in incs), s)
    open(CFG, "w").write(s)
    subprocess.run(["sudo", "systemctl", "restart", "lichessbot"], check=False)


def main():
    e = json.load(open(ETAT)) if os.path.exists(ETAT) else {"cat": 0, "essais": 0, "depuis": time.time()}
    maintenant = time.time()
    log = subprocess.run(["journalctl", "-u", "lichessbot", "--since", "@%d" % int(e["depuis"]),
                          "--no-pager", "-o", "cat"], capture_output=True, text=True).stdout
    essais = log.count("Challenging a random bot")
    parties = len(re.findall(r"Game over|Game won|Game lost|Game drawn|draw", log))
    e["depuis"] = maintenant
    if parties:
        e["essais"] = 0
    else:
        e["essais"] += essais
    if e["essais"] >= ESSAIS:
        e["cat"] = (e["cat"] + 1) % len(CATEGORIES)
        e["essais"] = 0
        appliquer(e["cat"])
        note(f"{ESSAIS} defis sans partie : passage au {CATEGORIES[e['cat']][0]}")
    json.dump(e, open(ETAT, "w"))


if __name__ == "__main__":
    main()
