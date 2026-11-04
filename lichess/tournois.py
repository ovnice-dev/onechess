"""OnechessBot et les tournois de bots, sans intervention (toutes les 5 minutes).

1. Inscrit le bot aux tournois a venir de ses equipes, et lit leur reglement
   (nom + description) pour y trouver une interdiction du livre d'ouvertures
   ou des tables de finales (anglais ou francais).
2. 30 minutes avant un tournoi : plus de defis (ni lances, ni acceptes), et
   coupe le livre ou les tables s'il les interdit ; 5 minutes apres sa fin,
   remet tout. Chaque changement redemarre lichess-bot.
3. Note chaque decision dans ~/tournois.log.
Le jeton est lu dans la configuration de lichess-bot ; il n'est jamais affiche.
"""
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

EQUIPES = ["lichess-bots", "bots--humans-league"]
ICI = os.path.expanduser("~")
CFG = os.path.join(ICI, "lichess-bot", "config.yml")
ETAT = os.path.join(ICI, "tournois_etat.json")
JOURNAL = os.path.join(ICI, "tournois.log")
TABLES = "/home/ubuntu/syzygy"
AVANT, APRES = 30 * 60, 5 * 60   # repos et reglages : 30 min avant le debut, 5 min apres la fin

NEG = r"(?:no|not|without|forbidden|banned|prohibited|disallowed|disabled|off|sans|pas\s+de|interdit\w*|aucun\w*)"
LIVRE = r"(?:opening\s*books?|books?|livres?(?:\s+d.ouvertures?)?|polyglot)"
TABLE = r"(?:table\s*bases?|tablebases?|syzygy|egtb|tb|endgame\s+tables?|tables?\s+de\s+finales?)"


def interdit(texte, objet):
    """Le reglement interdit-il l'objet (LIVRE ou TABLE) ? On decoupe le texte
    en propositions ; une negation dans la meme proposition que l'objet suffit
    (prudent : mieux vaut couper pour rien que d'etre disqualifie)."""
    for prop in re.split(r"[.,;!?\n]|\s(?:and|et|but|mais)\s", texte.lower()):
        if re.search(rf"\b{objet}\b", prop) and re.search(rf"\b{NEG}\b", prop):
            return True
    return False


def jeton():
    return re.search(r"^token:\s*\"?([^\"\n]+)\"?", open(CFG).read(), re.M).group(1)


def api(url, data=None, accept="application/json"):
    req = urllib.request.Request("https://lichess.org" + url, data=data,
                                 headers={"Authorization": "Bearer " + jeton(), "Accept": accept})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def note(msg):
    ligne = time.strftime("%d-%m %H:%M ") + msg
    print(ligne, flush=True)
    with open(JOURNAL, "a") as f:
        f.write(ligne + "\n")


def config_actuelle():
    """(livre, tables, ouvert) ; ouvert = defis automatiques et defis recus acceptes."""
    s = open(CFG).read()
    livre = re.search(r"^  polyglot:\n    enabled: (\w+)", s, re.M)
    tables = re.search(r"^    SyzygyPath: \"([^\"]*)\"", s, re.M)
    defis = re.search(r"^  allow_matchmaking: (\w+)", s, re.M)
    return (bool(livre) and livre.group(1) == "true",
            bool(tables) and tables.group(1) not in ("", "<empty>"),
            bool(defis) and defis.group(1) == "true")


def regler(livre, tables, ouvert):
    s = open(CFG).read()
    s = re.sub(r"^(  polyglot:\n    enabled: )\w+", r"\g<1>" + ("true" if livre else "false"), s, flags=re.M)
    s = re.sub(r"^    SyzygyPath: \"[^\"]*\"", "    SyzygyPath: \"" + (TABLES if tables else "<empty>") + "\"",
               s, flags=re.M)
    # Repos avant et pendant un tournoi : plus de defis lances, et les defis
    # recus sont refuses avec le motif "later" (0 partie par adversaire). Les
    # parties du tournoi ne sont pas des defis : elles ne sont pas touchees.
    s = re.sub(r"^  allow_matchmaking: \w+", "  allow_matchmaking: " + ("true" if ouvert else "false"),
               s, flags=re.M)
    s = re.sub(r"^  max_simultaneous_games_per_user: \d+\n", "", s, flags=re.M)
    if not ouvert:
        s = re.sub(r"^(challenge:\n)", r"\g<1>  max_simultaneous_games_per_user: 0\n", s, flags=re.M)
    open(CFG, "w").write(s)
    subprocess.run(["sudo", "systemctl", "restart", "lichessbot"], check=False)


def secondes(iso_ou_ms):
    if isinstance(iso_ou_ms, (int, float)):
        return iso_ou_ms / 1000
    return datetime.datetime.fromisoformat(iso_ou_ms.replace("Z", "+00:00")).timestamp()


def main():
    etat = json.load(open(ETAT)) if os.path.exists(ETAT) else {}
    if "--equipes" in sys.argv:
        for e in EQUIPES:
            try:
                api(f"/team/{e}/join", data=b"")
                note(f"equipe rejointe : {e}")
            except Exception as x:
                note(f"equipe {e} : {x}")

    # 1. Inscriptions et lecture des reglements
    for e in EQUIPES:
        try:
            lignes = api(f"/api/team/{e}/arena?max=20&status=created", accept="application/x-ndjson")
        except Exception as x:
            note(f"{e} : {x}")
            continue
        for l in lignes.splitlines():
            if not l.strip():
                continue
            t = json.loads(l)
            if t["id"] in etat:
                continue
            try:
                d = json.loads(api(f"/api/tournament/{t['id']}"))
                texte = (d.get("fullName") or "") + ". " + (d.get("description") or "")
                debut = secondes(d.get("startsAt", t.get("startsAt")))
                fin = debut + 60 * int(d.get("minutes", t.get("minutes", 60)))
                regle = {"nom": d.get("fullName", ""), "debut": debut, "fin": fin,
                         "livre": not interdit(texte, LIVRE), "tables": not interdit(texte, TABLE)}
                api(f"/api/tournament/{t['id']}/join", data=b"")
                etat[t["id"]] = regle
                note(f"inscrit : {regle['nom']} ({t['id']}), le "
                     f"{time.strftime('%d-%m %H:%M', time.gmtime(debut))} UTC ; livre "
                     f"{'permis' if regle['livre'] else 'INTERDIT'}, tables "
                     f"{'permises' if regle['tables'] else 'INTERDITES'}")
            except Exception as x:
                note(f"inscription refusee : {t.get('fullName')} ({t['id']}) : {x}")

    # 2. Reglage du bot selon les tournois en cours ou imminents
    maintenant = time.time()
    actifs = [r for r in etat.values() if r["debut"] - AVANT <= maintenant <= r["fin"] + APRES]
    voulu = (all(r["livre"] for r in actifs), all(r["tables"] for r in actifs), not actifs)
    if voulu != config_actuelle():
        regler(*voulu)
        note(f"reglage : livre {'oui' if voulu[0] else 'non'}, tables {'oui' if voulu[1] else 'non'}, "
             f"defis {'oui' if voulu[2] else 'non (repos tournoi)'}"
             + (f" (tournoi : {', '.join(r['nom'] for r in actifs)})" if actifs else " (plus de tournoi en cours)"))

    # Les tournois finis depuis plus d'un jour sortent de l'etat
    etat = {k: r for k, r in etat.items() if r["fin"] > maintenant - 86400}
    json.dump(etat, open(ETAT, "w"), indent=1)


if __name__ == "__main__":
    if "--test" in sys.argv:
        for txt in ["No opening books allowed. Tablebases OK.", "Books are not allowed", "Without tablebases please",
                    "Syzygy forbidden, books allowed", "Bots welcome! 3+2 arena", "Sans livre d'ouverture",
                    "Tablebases are allowed", "No TB, no book", "This tournament is organized by the team X."]:
            print(f"{txt!r:45} livre {'INTERDIT' if interdit(txt, LIVRE) else 'ok':9} "
                  f"tables {'INTERDITES' if interdit(txt, TABLE) else 'ok'}")
    else:
        main()
