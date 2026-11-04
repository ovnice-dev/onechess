# Onechess sur Lichess

Ce que l'utilisateur doit faire lui-même (création de compte et jeton) :

1. Créer un **nouveau** compte Lichess qui n'a joué **aucune** partie (un compte
   BOT ne peut plus jouer comme un humain, et un compte qui a déjà joué ne peut
   pas devenir BOT).
2. Créer un jeton d'API sur https://lichess.org/account/oauth/token avec la
   permission `bot:play`.
3. Passer le compte en BOT (une seule fois, irréversible), avec ce jeton :
   `curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer <jeton>"`
4. Installer lichess-bot (`git clone https://github.com/lichess-bot-devs/lichess-bot`,
   puis `pip install -r requirements.txt`), copier `config.yml.modele` en
   `config.yml`, y mettre le jeton, et placer l'exécutable et `onechess.nnue`
   dans `engines/`.
5. Lancer : `python lichess-bot.py` (Python 3.11 ou plus récent).

Le jeton est un secret : ne jamais le publier ni le copier dans le dépôt.

## Tournois (automatique)

`tournois.py`, lancé toutes les 5 minutes (minuterie systemd), inscrit le bot
aux tournois à venir de ses équipes de bots, lit leur règlement et coupe le
livre ou les tables 15 minutes avant un tournoi qui les interdit, puis remet
tout 5 minutes après la fin. `regles.sh` permet de le faire à la main.
