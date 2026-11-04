# Onechess

*[English version](README.md)*

Moteur d'échecs UCI écrit en C++, avec une évaluation par réseau de neurones NNUE.

**Auteur et créateur : ovnice** (GitHub : [ovnice-dev](https://github.com/ovnice-dev)).
Onechess a été conçu et développé à partir de 2026 par ovnice ; l'historique
de ce dépôt et le dépôt d'archive privé en gardent la trace datée.

## Caractéristiques

- recherche alpha-bêta (PVS) avec table de transposition, LMR, coup nul,
  historiques de continuation et de menaces, corrections d'évaluation ;
- réseau NNUE 768 → 512×2 → 4 sorties (4 experts selon le matériel), entraîné
  sur des parties Lichess Elite étiquetées et sur de l'autojeu Stockfish 19 ;
- tables de finales Syzygy (via Fathom) ;
- gestion du temps par effort en nœuds ;
- UCI : ponder, Chess960, `go mate`, Syzygy ;
- chaque changement est validé par SPRT pentanomial (ouvertures UHO).

## Jouer contre Onechess

Onechess joue sur Lichess sous le nom [OnechessBot](https://lichess.org/@/OnechessBot).

## Compiler

```bash
cd engine
bash build.sh
```

## Utiliser

Placer `onechess.exe` et `onechess.nnue` dans le même dossier, puis ajouter
`onechess.exe` comme moteur UCI dans l'interface (En Croissant, Arena, Cute
Chess…). Le réseau est cherché à côté de l'exécutable ; sans lui, le moteur
joue avec une évaluation simple, beaucoup plus faible.

## Options UCI

| Option | Rôle |
|---|---|
| `Hash`, `Threads`, `Clear Hash` | table de transposition (Mo), fils de recherche |
| `Move Overhead` | marge de temps par coup (ms), pour les interfaces lentes |
| `EvalFile` | réseau NNUE (par défaut `onechess.nnue`, cherché à côté de l'exécutable) |
| `Ponder`, `UCI_Chess960` | réflexion sur le temps adverse, échecs aléatoires Fischer |
| `SyzygyPath`, `SyzygyProbeDepth`, `SyzygyProbeLimit`, `Syzygy50MoveRule` | tables de finales Syzygy |
| `MultiPV` | nombre de meilleures lignes affichées (analyse) |
| `UCI_ShowWDL` | probabilités gain / nulle / perte dans chaque ligne `info` |
| `Debug Log File` | copie de tout le dialogue UCI dans un fichier |

Les constantes internes de la recherche restent réglables par `setoption`
(tests et réglages), mais ne sont pas listées par `uci`.

## Quel binaire choisir

| Binaire | Processeurs |
|---|---|
| `onechess-bmi2.exe` | Intel depuis Haswell (2013), AMD Zen 3 et suivants |
| `onechess-avx2.exe` | AMD Zen 1 et Zen 2 |
| `onechess-sse41.exe` | processeurs sans AVX2 (lent) |

Les trois jouent exactement la même partie (`bench 12` identique) ; seule la
vitesse change. Ils se construisent par `bash tools/diffusion.sh`.

## Données d'entraînement

- Parties de la base Lichess Elite, positions étiquetées par Stockfish.
- Autojeu de Stockfish 19 produit pour ce projet.
- Autojeu public de Stockfish à 5 000 nœuds (`official-stockfish/master-binpacks`
  sur Hugging Face, licence ODbL), pour les réseaux à venir.
- Lecture des fichiers binpack : code de `nnue-pytorch` (GPLv3), dans
  `tools/binpack/lib/`.

## Licence

Copyright (C) 2026 ovnice.

Onechess est un logiciel libre distribué sous la **GNU General Public License
version 3** (voir [LICENSE](LICENSE)). Vous pouvez l'utiliser, le modifier et le
redistribuer, **à condition** de conserver cette mention de copyright et le
crédit de l'auteur, de fournir le code source de toute version distribuée, et
de la placer sous la même licence.

Toute version dérivée doit indiquer clairement qu'elle est dérivée
d'Onechess, créé par ovnice.

Composants tiers : Fathom (`engine/fathom/`), licence MIT, voir
`engine/fathom/LICENSE`.
