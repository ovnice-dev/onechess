# Onechess

*[Version française](README.fr.md)*

Onechess is a free UCI chess engine written in C++, with an NNUE evaluation
trained from scratch by its author. It plays standard chess and Chess960, and
can be used in any UCI interface (En Croissant, Cute Chess, Arena, Banksia…).

**Author: ovnice** ([ovnice-dev](https://github.com/ovnice-dev)), since 2026.

## Download

Get the latest version from the [Releases](https://github.com/ovnice-dev/onechess/releases)
page. Each release contains the executables and the network `onechess.nnue`:
put the executable and the network **in the same folder**.

| Executable | Processors |
|---|---|
| `onechess-1.0-bmi2.exe` | Intel since Haswell (2013), AMD Zen 3 and later (**the fastest**) |
| `onechess-1.0-avx2.exe` | AMD Zen 1 and Zen 2 |
| `onechess-1.0-sse41.exe` | older processors without AVX2 (slow) |

All executables play exactly the same moves (same `bench` signature); only the
speed differs. If unsure, try `bmi2` first: if it does not start, use `avx2`.

## Strength

Onechess plays on Lichess as [OnechessBot](https://lichess.org/@/OnechessBot).
Our own estimate for version 1.0, from a 1,500-game gauntlet against five engines
rated around 3600 (10+0.1, one core each): **about 3577 on the CCRL 40/15
single-core scale** (November 2026). Official rating lists results will be
added here once published.

## Features

**Search**: principal variation search with aspiration windows, iterative
deepening and a shared transposition table; null move, reverse futility,
razoring, ProbCut, late move reductions and pruning, singular, negative and
check extensions; butterfly, capture, continuation, pawn and threat-aware
histories; evaluation correction histories (pawn, non-pawn, minor, major,
continuation); Lazy SMP for several threads.

**Evaluation**: an NNUE network with 768 piece-square inputs × 4 experts chosen by material,
a 1024-neuron feature transformer per side, 4 output buckets. Trained by its
author on 1.77 billion positions of public Stockfish self-play data
(`official-stockfish/master-binpacks`, ODbL), with an in-house trainer
(PyTorch on GPU, JAX on TPU).

**Endgames**: Syzygy tablebases (through Fathom), at the root and in search.

**Time management**: node-effort based, for increment, sudden death and
moves-to-go controls; ponder.

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

Other UCI commands: `bench` (signature and speed), `eval`, `d`, `perft`.

## Build from source

```bash
cd engine
./build.sh pgo        # profile-guided build, the fastest
```

Requires a C++20 compiler (GCC or Clang). Check the build with
`bench 12`: the node count must match the signature given in the release notes.

## How it is developed

Every change is measured before it is kept: time-control SPRT matches
(pentanomial, UHO openings) on many machines, NNUE training on Kaggle GPUs and
TPUs. The testing and training tools live in a separate repository:
[onechess-tools](https://github.com/ovnice-dev/onechess-tools).

## License

Copyright (C) 2026 ovnice.

Onechess is free software released under the **GNU General Public License
version 3** (see [LICENSE](LICENSE)). You may use, modify and redistribute it,
provided that you keep this copyright notice and the author's credit, provide
the source code of any distributed version, and release it under the same
license. Any derived version must clearly state that it is derived from
Onechess, created by ovnice.

Third-party code: Fathom (`engine/fathom/`, MIT license). Ideas from many open
engines, Stockfish first, are credited in the source comments.
