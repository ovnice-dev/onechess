# Onechess

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
