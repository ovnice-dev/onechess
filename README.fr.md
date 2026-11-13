# Onechess

*[English version](README.md)*

Onechess est un moteur d'échecs UCI libre, écrit en C++, avec une évaluation
NNUE entraînée de zéro par son auteur. Il joue aux échecs classiques et au
Chess960, et s'utilise dans toute interface UCI (En Croissant, Cute Chess,
Arena, Banksia…).

**Auteur : ovnice** ([ovnice-dev](https://github.com/ovnice-dev)), depuis 2026.

## Téléchargement

La dernière version est sur la page [Releases](https://github.com/ovnice-dev/onechess/releases).
Chaque version contient les exécutables et le réseau `onechess.nnue` : mettre
l'exécutable et le réseau **dans le même dossier**.

| Exécutable | Processeurs |
|---|---|
| `onechess-1.0-bmi2.exe` | Intel depuis Haswell (2013), AMD Zen 3 et suivants (**le plus rapide**) |
| `onechess-1.0-avx2.exe` | AMD Zen 1 et Zen 2 |
| `onechess-1.0-sse41.exe` | processeurs anciens sans AVX2 (lent) |

Tous jouent exactement les mêmes coups (même signature `bench`) ; seule la
vitesse change. Dans le doute, essayer `bmi2` ; s'il ne démarre pas, `avx2`.

## Force

Onechess joue sur Lichess sous le nom [OnechessBot](https://lichess.org/@/OnechessBot).
Notre propre estimation pour la version 1.0, d'après un tournoi de 1 500 parties
contre cinq moteurs notés autour de 3 600 (10+0,1, un cœur chacun) : **environ
3 577 sur l'échelle CCRL 40/15 mono-cœur** (novembre 2026). Les résultats des
listes officielles seront ajoutés ici dès leur publication.

## Ce qu'il sait faire

**Recherche** : PVS avec fenêtres d'aspiration, approfondissement itératif,
table de transposition partagée ; coup nul, futilité inverse, razoring,
ProbCut, réductions et élagages des coups tardifs, extensions singulières,
négatives et d'échec ; historiques (papillon, prises, continuation, pions,
menaces) ; historiques de correction de l'évaluation ; Lazy SMP à plusieurs fils.

**Évaluation** : un réseau NNUE à 768 entrées pièce-case × 4 experts choisis par le
matériel, 1024 neurones par camp, 4 sorties. Entraînée par l'auteur sur
1,77 milliard de positions d'autojeu public de Stockfish
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
| `EvalFile` | réseau NNUE (défaut `onechess.nnue`, cherché à côté de l'exécutable) |
| `Ponder`, `UCI_Chess960` | réflexion sur le temps adverse, échecs aléatoires Fischer |
| `SyzygyPath`, `SyzygyProbeDepth`, `SyzygyProbeLimit`, `Syzygy50MoveRule` | tables de finales Syzygy |
| `MultiPV` | nombre de meilleures lignes affichées (analyse) |
| `UCI_ShowWDL` | probabilités victoire / nulle / défaite dans chaque ligne `info` |
| `Debug Log File` | copie de tout le dialogue UCI dans un fichier |

Autres commandes : `bench` (signature et vitesse), `eval`, `d`, `perft`.

## Compiler

```bash
cd engine
./build.sh pgo        # compilation guidée par profil, la plus rapide
```

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

Code tiers : Fathom (`engine/fathom/`, licence MIT). Les idées reprises des
moteurs libres, Stockfish en tête, sont créditées dans les commentaires du code.
