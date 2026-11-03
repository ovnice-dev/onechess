# Scripts de reproduction (revue Onechess du 02-11-2026)

Variables d'environnement communes :

- `ONECHESS` : racine du depot (par defaut `.`)
- `SYZYGY` : dossier de tables Syzygy (les tables 3-4 pieces du depot
  python-chess suffisent : `data/syzygy/regular`)

Dependances Python : `pip install chess numpy`.

| Script | Ce qu'il verifie | Exemple |
|---|---|---|
| `keycheck.cpp` | Parties aleatoires (classique + Chess960) : cle Zobrist, cles pions et non-pions, materiel, echecs, clouages, aller-retour FEN, `undo_move`, `gives_check`, `pseudo_legal` face a des coups corrompus | `g++ -std=c++20 -O1 -g -march=native -pthread -I$ONECHESS/engine/src -I$ONECHESS/engine/fathom keycheck.cpp $(ls $ONECHESS/engine/src/*.cpp \| grep -v main.cpp) $ONECHESS/engine/fathom/tbprobe.c -o keycheck && ./keycheck 10000` |
| `perft_diff.py` | Perft du moteur contre python-chess sur positions tirees au hasard (moitie Chess960) | `python3 perft_diff.py ./onechess 300 3` |
| `nnue_diff.py` | Evaluation du moteur contre la reference numpy de `tools/check_nnue.py`, sur N positions | `python3 nnue_diff.py ./onechess reseau/onechess.nnue 200` |
| `synth_net.py` | Reseaux synthetiques au format du moteur (couches cachees, pairmul, paires de pions, buckets de roi) | `python3 synth_net.py nets/s` |
| `stress.py` | Recherches courtes en serie sur la version de debogage (assertions actives) | `python3 stress.py ./onechess-debug 600 2 $SYZYGY 5` |
| `tb_scores.py` | Compte les scores aberrants affiches avec les tables chargees (defaut 1) | `python3 tb_scores.py ./onechess 300` |
| `nodes_dist.sh` | Depassement de `go nodes N` sur les positions du banc (defaut 4) | `bash nodes_dist.sh ./onechess 25000` |
| `uci_timing.py` | Horodatage des reponses UCI (defaut 7, ponder) | `python3 uci_timing.py ./onechess "position startpos" "go ponder movetime 300" "sleep:2" "stop"` |
