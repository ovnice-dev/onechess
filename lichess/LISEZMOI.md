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
livre ou les tables (et arrete les defis) 30 minutes avant un tournoi, puis remet
tout 5 minutes après la fin. `regles.sh` permet de le faire à la main.

## Mise à jour du moteur (automatique)

`maj_bot.py`, lancé toutes les heures, regarde la dernière release GitHub
(étiquette, `onechess.nnue`, `signature.txt`), compile le moteur à cette
étiquette, vérifie la signature `bench 12`, attend la fin de la partie en cours
et redémarre lichess-bot (qui garde son moteur d'une partie à l'autre).

## Garde-fou (automatique)

`garde.py`, toutes les 5 minutes : si lichess-bot croit jouer une partie que
Lichess ne voit pas (deux fois de suite), il redémarre lichess-bot. Le 6-11, une
partie perdue au temps avait bloqué le bot 10 heures.

## Cadences des défis (automatique)

`categories.py`, lancé chaque minute, fait tourner la cadence des défis
automatiques : blitz, puis rapide, dès que 10 défis de suite n'ont
donné aucune partie. La catégorie qui donne des parties est gardée. Pas de bullet : Lichess limite
un bot à 100 parties par jour contre d'autres bots.
