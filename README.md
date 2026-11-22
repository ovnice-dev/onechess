# Onechess

*[Version française](README.fr.md)*

Onechess is a free UCI chess engine written in C++, with its own NNUE
evaluation (architecture, trainer and networks by its author). It plays standard chess and Chess960, and
can be used in any UCI interface (En Croissant, Cute Chess, Arena, Banksia…).

**Author: ovnice** ([ovnice-dev](https://github.com/ovnice-dev)), since 2026.

## Download

Get the latest version from the [Releases](https://github.com/ovnice-dev/onechess/releases)
page. Each release contains the executables and the network `onechess.nnue`:
put the executable and the network **in the same folder**.

| Executable | Processors |
|---|---|
| `onechess-1.2-avx512.exe` | recent Intel and AMD processors with AVX-512 |
| `onechess-1.2-bmi2.exe` | Intel since Haswell (2013), AMD Zen 3 and later |
| `onechess-1.2-avx2.exe` | AMD Zen 1 and Zen 2 |
| `onechess-1.2-sse41.exe` | older processors without AVX2 (slow) |
| `onechess-1.2-arm64-musl` | 64-bit ARM: Android phones and tablets, ARM Linux (static, NEON) |

All executables play exactly the same moves (same `bench` signature); only the
speed differs. If unsure, try `avx512`, then `bmi2`, then `avx2`, and keep the
fastest one that starts (compare the `nps` given by `bench`). On Android, if the
interface does not find the network, set `EvalFile` to the full path of
`onechess.nnue`.

## Strength

Onechess plays on Lichess as [OnechessBot](https://lichess.org/@/OnechessBot).
Our own measurement for version 1.0, from a 1,500-game gauntlet against five engines
rated around 3600 (10+0.1, one core each): **about 3577 on the CCRL 40/15
single-core scale** (November 2026). Version 1.2 is about 30 % faster with a
retuned search: our estimate is **about 3590**. Official rating lists results
will be added here once published.

## Features

**Search**: principal variation search with aspiration windows, iterative
deepening and a shared transposition table; null move, reverse futility,
razoring, ProbCut, late move reductions and pruning, singular, negative and
check extensions; butterfly, capture, continuation, pawn and threat-aware
histories; evaluation correction histories (pawn, non-pawn, minor, major,
continuation); Lazy SMP for several threads.

**Evaluation**: an NNUE network with 768 piece-square inputs × 4 experts chosen by material,
a 1024-neuron feature transformer per side, 4 output buckets, with incremental
updates in AVX-512, AVX2 or NEON. Trained by its
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

Other UCI commands: `bench` (signature and speed), `eval`, `d`, `perft` (also `go perft N`).

## Build from source

```bash
cd engine
./build.sh clang      # Clang with profile-guided optimisation, the fastest
./build.sh pgo        # the same with GCC
```

On ARM64 (Linux, Android with a native compiler): `g++ -std=c++20 -O3 -DNDEBUG
-DONE_FIGE -flto=auto -march=armv8-a -pthread -Ifathom src/*.cpp
fathom/tbprobe.c -o onechess` from the `engine` folder.

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

Credits:
- Onechess is an independent engine. Its code is original, written for
  Onechess by its author.
- Its NNUE evaluation (architecture with material experts, inference code,
  trainers, networks) is the author's own work.
- Search ideas come from the chess programming community and many open
  engines (Stockfish, Ethereal, Berserk, Koivisto, PlentyChess, Reckless and
  others); they are credited in the source comments.
- Third-party code: **Fathom** (`engine/fathom/`, MIT license) for Syzygy
  tablebase probing.
- Training data: public Stockfish self-play data (ODbL).
