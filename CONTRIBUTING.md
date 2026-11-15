# Contributing to Onechess

Thank you for your interest in Onechess!

## Reporting a problem

Open an issue with the bug report template. Please include:

- the Onechess version (`uci` prints it) and the executable you use (`bmi2`, `avx2` or `sse41`);
- the output of `bench 12` (the node count is the signature of the engine);
- the interface you use and, if possible, the position (FEN) and the UCI commands that show the problem.

## Proposing a change

Onechess only keeps changes that are **measured**:

- a search or evaluation change must win a time-control SPRT (pentanomial
  statistics, UHO openings, 10+0.1) before it is merged;
- a speed change must keep the `bench 12` signature and be measured with
  alternating runs (ABBA);
- a change that is not meant to alter play must keep the `bench 12` signature.

Open a pull request with the template: describe the idea, give the test result
(Elo, error bar, number of games) and the new `bench 12` signature if it changes.

## License

By contributing, you agree that your contribution is released under the
GNU General Public License version 3, like the rest of Onechess.
