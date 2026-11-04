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
5. Lancer : `python lichess-bot.py`.

Le jeton est un secret : ne jamais le publier ni le copier dans le dépôt.
