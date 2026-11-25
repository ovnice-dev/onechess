# Onechess

*[English version](README.md)*

Onechess est un moteur d'échecs UCI libre, écrit en C++, avec sa propre
évaluation NNUE (architecture, entraîneur et réseaux de son auteur). Il joue aux échecs classiques et au
Chess960, et s'utilise dans toute interface UCI (En Croissant, Cute Chess,
Arena, Banksia…).

**Auteur : ovnice** ([ovnice-dev](https://github.com/ovnice-dev)), depuis 2026.

## Téléchargement

La dernière version est sur la page [Releases](https://github.com/ovnice-dev/onechess/releases).
Depuis la version 1.2.1, le réseau est **intégré à chaque exécutable** : il
suffit de lancer l'exécutable, rien d'autre n'est nécessaire. Le fichier
`onechess.nnue` reste fourni ; s'il est placé à côté de l'exécutable (ou choisi
avec `EvalFile`), il remplace le réseau intégré.

| Exécutable | Processeurs |
|---|---|
| `onechess-1.2.1-avx512.exe` | processeurs Intel et AMD récents avec AVX-512 |
| `onechess-1.2.1-bmi2.exe` | Intel depuis Haswell (2013), AMD Zen 3 et suivants |
| `onechess-1.2.1-avx2.exe` | AMD Zen 1 et Zen 2 |
| `onechess-1.2.1-sse41.exe` | processeurs anciens sans AVX2 (lent) |
| `onechess-1.2.1-arm64-musl` | ARM 64 bits : téléphones et tablettes Android, Linux ARM (statique, NEON) |

Tous jouent exactement les mêmes coups (même signature `bench`) ; seule la
vitesse change. Dans le doute, essayer `avx512`, puis `bmi2`, puis `avx2`, et
garder le plus rapide qui démarre (comparer les `nps` de `bench`).

## Force

Onechess joue sur Lichess sous le nom [OnechessBot](https://lichess.org/@/OnechessBot).
Notre propre mesure pour la version 1.0, d'après un tournoi de 1 500 parties
contre cinq moteurs notés autour de 3 600 (10+0,1, un cœur chacun) : **environ
3 577 sur l'échelle CCRL 40/15 mono-cœur** (novembre 2026). La version 1.2 est
environ 30 % plus rapide, avec une recherche re-réglée : notre estimation est
**environ 3 590**. La version 1.2.1 ajoute une gestion du temps re-réglée, des
corrections d'évaluation plus rapides, et ne perd plus de vitesse quand les
tables Syzygy sont chargées. Les résultats des listes officielles seront ajoutés
ici dès leur publication.

## Ce qu'il sait faire

**Recherche** : PVS avec fenêtres d'aspiration, approfondissement itératif,
table de transposition partagée ; coup nul, futilité inverse, razoring,
ProbCut, réductions et élagages des coups tardifs, extensions singulières,
négatives et d'échec ; historiques (papillon, prises, continuation, pions,
menaces) ; historiques de correction de l'évaluation ; Lazy SMP à plusieurs fils.

**Évaluation** : un réseau NNUE à 768 entrées pièce-case × 4 experts choisis par le
matériel, 1024 neurones par camp, 4 sorties, mises à jour incrémentales en
AVX-512, AVX2 ou NEON. Entraînée par l'auteur sur
plusieurs milliards de positions d'autojeu public de Stockfish
(`official-stockfish/master-binpacks`, ODbL), avec un entraîneur maison
(PyTorch sur GPU, JAX sur TPU).

**Finales** : tables Syzygy (par Fathom), à la racine et dans la recherche.

**Gestion du temps** : fondée sur l'effort par coup, pour l'incrément, la mort
subite et le nombre de coups imposé ; ponder.

## Options UCI

| Option | Rôle |
|---|---|
| `Hash`, `Threads`, `Clear Hash` | table de transposition (Mo), fils de recherche |
| `Move Overhead` | marge de temps par coup (ms), pour les interfaces lentes |
| `EvalFile` | fichier du réseau NNUE (défaut `onechess.nnue`, cherché à côté de l'exécutable ; absent, le réseau intégré est utilisé) |
| `Ponder`, `UCI_Chess960` | réflexion sur le temps adverse, échecs aléatoires Fischer |
| `SyzygyPath`, `SyzygyProbeDepth`, `SyzygyProbeLimit`, `Syzygy50MoveRule` | tables de finales Syzygy |
| `MultiPV` | nombre de meilleures lignes affichées (analyse) |
| `UCI_ShowWDL` | probabilités victoire / nulle / défaite dans chaque ligne `info` |
| `Debug Log File` | copie de tout le dialogue UCI dans un fichier |

Autres commandes : `bench` (signature et vitesse), `eval`, `d`, `perft` (et `go perft N`).

## Compiler

```bash
cd engine
./build.sh clang      # Clang guidé par profil, le plus rapide
./build.sh pgo        # la même chose avec GCC
```

Sur ARM64 (Linux, Android avec un compilateur natif), depuis le dossier
`engine` : `g++ -std=c++20 -O3 -DNDEBUG -DONE_FIGE -flto=auto -march=armv8-a
-pthread -Ifathom src/*.cpp fathom/tbprobe.c -o onechess`.

Un compilateur C++20 (GCC ou Clang) suffit. Vérifier avec `bench 12` : le
nombre de nœuds doit être la signature donnée dans les notes de la version.

## Comment il est développé

Chaque changement est mesuré avant d'être gardé : matchs SPRT au temps
(pentanomial, ouvertures UHO) sur de nombreuses machines, entraînements NNUE
sur les GPU et TPU de Kaggle. Les outils de test et d'entraînement sont dans un
dépôt à part : [onechess-tools](https://github.com/ovnice-dev/onechess-tools).

## Licence

Copyright (C) 2026 ovnice.

Onechess est un logiciel libre sous **licence publique générale GNU version 3**
(voir [LICENSE](LICENSE)). Vous pouvez l'utiliser, le modifier et le
redistribuer, à condition de conserver cette mention et le crédit de l'auteur,
de fournir le code source de toute version distribuée, et de la publier sous la
même licence. Toute version dérivée doit indiquer clairement qu'elle dérive
d'Onechess, créé par ovnice.

Crédits :
- Onechess est un moteur indépendant. Son code est original, écrit pour
  Onechess par son auteur.
- Son évaluation NNUE (architecture à experts par matériel, code
  d'inférence, entraîneurs, réseaux) est le travail de son auteur.
- Les idées de recherche viennent de la communauté de la programmation
  d'échecs et de nombreux moteurs libres (Stockfish, Ethereal, Berserk,
  Koivisto, PlentyChess, Reckless et d'autres) ; elles sont citées dans les
  commentaires du code.
- Code tiers : **Fathom** (`engine/fathom/`, licence MIT), sondage des tables
  Syzygy.
- Données d'entraînement : données publiques d'autojeu de Stockfish (ODbL).
