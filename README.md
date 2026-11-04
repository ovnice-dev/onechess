# Onechess

*[Version française](README.fr.md)*

UCI chess engine written in C++, with an NNUE neural-network evaluation.

**Author and creator: ovnice** (GitHub: [ovnice-dev](https://github.com/ovnice-dev)).
Onechess has been designed and developed by ovnice since 2026; the history of
this repository and of the private archive repository keep a dated record of it.

## Features

- alpha-beta search (PVS) with transposition table, LMR, null move,
  continuation and threat histories, evaluation corrections;
- NNUE network 768 → 512×2 → 4 outputs (4 experts selected by material),
  trained on labelled Lichess Elite games and Stockfish 19 self-play;
- Syzygy endgame tablebases (through Fathom);
- node-effort based time management;
- UCI: ponder, Chess960, `go mate`, MultiPV, WDL output, Syzygy;
- every change is validated by a pentanomial SPRT (UHO openings).

## Play against Onechess

Onechess plays on Lichess as [OnechessBot](https://lichess.org/@/OnechessBot).

## Build

```bash
cd engine
bash build.sh
```

## Usage

Put `onechess.exe` and `onechess.nnue` in the same folder, then add
`onechess.exe` as a UCI engine in your GUI (En Croissant, Arena, Cute Chess…).
The network is looked up next to the executable; without it, the engine falls
back to a simple evaluation and is much weaker.

## UCI options

| Option | Purpose |
|---|---|
| `Hash`, `Threads`, `Clear Hash` | transposition table (MB), search threads |
| `Move Overhead` | time margin per move (ms), for slow interfaces |
| `EvalFile` | NNUE network (default `onechess.nnue`, looked up next to the executable) |
| `Ponder`, `UCI_Chess960` | thinking on the opponent's time, Fischer random chess |
| `SyzygyPath`, `SyzygyProbeDepth`, `SyzygyProbeLimit`, `Syzygy50MoveRule` | Syzygy endgame tablebases |
| `MultiPV` | number of best lines shown (analysis) |
| `UCI_ShowWDL` | win / draw / loss probabilities in each `info` line |
| `Debug Log File` | copy of the whole UCI dialogue into a file |

The internal search constants can still be set with `setoption` (for testing
and tuning), but they are not listed by `uci`.

## Which binary to choose

| Binary | Processors |
|---|---|
| `onechess-bmi2.exe` | Intel since Haswell (2013), AMD Zen 3 and later |
| `onechess-avx2.exe` | AMD Zen 1 and Zen 2 |
| `onechess-sse41.exe` | processors without AVX2 (slow) |

All three play exactly the same game (identical `bench 12`); only the speed
differs. They are built with `bash tools/diffusion.sh`.

## Training data

- Games from the Lichess Elite database, positions labelled by Stockfish.
- Stockfish 19 self-play generated for this project.
- Public Stockfish self-play at 5,000 nodes (`official-stockfish/master-binpacks`
  on Hugging Face, ODbL license), for the upcoming networks.
- Binpack reading: code from `nnue-pytorch` (GPLv3), in `tools/binpack/lib/`.

## License

Copyright (C) 2026 ovnice.

Onechess is free software distributed under the **GNU General Public License
version 3** (see [LICENSE](LICENSE)). You may use, modify and redistribute it,
**provided that** you keep this copyright notice and the author's credit,
provide the source code of any distributed version, and release it under the
same license.

Any derived version must clearly state that it is derived from Onechess,
created by ovnice.

Third-party components: Fathom (`engine/fathom/`), MIT license, see
`engine/fathom/LICENSE`.
