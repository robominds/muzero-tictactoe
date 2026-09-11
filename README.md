# MuZero-Style Tic-Tac-Toe

A from-scratch, dependency-free C++17 implementation of MuZero — self-play
guided by PUCT/MCTS over a **learned** model, with a hand-written neural
network (forward pass and backpropagation written by hand, no autograd
library) — applied to tic-tac-toe as a learning project and as a direct,
matched comparison against the sibling
[`alphazero-tictactoe`](../alphazero-tictactoe) implementation. The central
idea: search still runs, but it plans inside a model the system learned for
itself rather than the real rules — the actual `Board` is consulted exactly
once per search, at the root, and everywhere below that the tree walks a
network's own imagination.

See
[`docs/superpowers/specs/2026-09-10-muzero-tictactoe-design.md`](docs/superpowers/specs/2026-09-10-muzero-tictactoe-design.md)
for the design rationale and
[`docs/superpowers/plans/2026-09-10-muzero-tictactoe.md`](docs/superpowers/plans/2026-09-10-muzero-tictactoe.md)
for the implementation plan.

For a guided walkthrough of the algorithm itself — the three functions,
action encoding, search over learned state, credit assignment, targets, and
training, with real code and RL fundamentals explained along the way — see
[`docs/algorithm-explained.md`](docs/algorithm-explained.md) (plain
markdown) or open [`docs/algorithm-explained.html`](docs/algorithm-explained.html)
in a browser for the illustrated version with diagrams.
[`docs/muzero-vs-alphazero.md`](docs/muzero-vs-alphazero.md) collects every
place the two projects diverge into a standalone comparison, and
[`docs/results.md`](docs/results.md) is the full, unfiltered account of what
was measured — including the shortfall described below.

## Build

Requires CMake 3.16+ and a C++17 compiler. No external dependencies.

```sh
mkdir build && cd build
cmake ..
cmake --build .
ctest --output-on-failure   # run the test suite (10 tests)
```

## Usage

Five executables are produced in `build/`: three programs and two
diagnostic tools.

```sh
# Train from scratch (or resume). Periodically prints self-play/training
# progress and evaluates against a perfect minimax player; saves a
# checkpoint at the end.
./train [iterations] [checkpoint-path] [seed] [resume-from]
# defaults: 400 iterations, checkpoint.bin, a random seed, fresh weights
#
# resume-from, if given, is an existing checkpoint to load weights from
# instead of starting from a fresh random init. It is weights only --
# trajectories are never checkpointed, so a resumed run starts with an
# empty replay buffer and spends its first iterations refilling it before
# training looks like it did before the interruption. The seed still fully
# determines everything from that point on, so resuming twice from the same
# checkpoint with the same seed reproduces the same run -- but a resumed
# run is NOT identical to an uninterrupted one of the same total length. A
# missing or corrupt checkpoint fails fast with an error rather than
# silently training from noise.

# Score a saved checkpoint against perfect minimax play (as both X and O).
./evaluate <checkpoint-path> [games-per-side]
# default: 50 games per side

# Play interactively against a saved checkpoint (you are X).
./play_cli <checkpoint-path>
```

Typical session, reproducing the one run that converged (see below):

```sh
./train 800 checkpoint.bin 303      # the third argument is the seed
./evaluate checkpoint.bin 50
./play_cli checkpoint.bin
```

Omit the seed and, on the evidence in `docs/results.md`, you have roughly
one-in-three odds of reproducing that result.

### What "trained" looks like

**The spec's first success criterion — converging to drawing play against
minimax — was not met.** Across 23 training runs at various
configurations, the best configuration found
(`numSimulations=100`, `temperatureMoves=6`, 800 iterations) converged on
**one of three seeds tested**, reaching `wins=0 draws=100 losses=0`
against perfect minimax play. The other two seeds at that identical
configuration ended at `draws=50 losses=50` and `draws=0 losses=100`
respectively, and no configuration tried converged on all three seeds of
any cell. For comparison, the sibling AlphaZero project reaches
`draws=40 losses=0` after just 20 iterations — though that is a single
reported run, not a seed grid, so the two results are not measured to the
same standard (see `docs/muzero-vs-alphazero.md`).

That one converged run is not nothing: it is direct, genuine evidence that
this implementation reaches optimal play end to end — search, the learned
model, and training all correct together, on at least one seed. What it
lacks is reliability, not soundness. The diagnosis in `docs/results.md`
localizes the failure to the policy head not reliably learning one rare
tactical pattern (blocking an immediate three-in-a-row), compounded by a
search tree that runs 18–19 plies deep on a 9-ply game with no legality
oracle below the root — most of the search budget is spent in positions no
legal game could still be in. Read
[`docs/results.md`](docs/results.md) for the full measurement, including
the `diag_eval` transcripts of the exact losing line, and
[`docs/muzero-vs-alphazero.md`](docs/muzero-vs-alphazero.md) for why this
gap is expected in kind (MuZero is doing structurally harder work) even
though its size here was not.

**The spec's second success criterion — `latent_probe` showing rolled-forward
dynamics staying close to a fresh representation pass — was also not met**;
see the `latent_probe` measurements in `docs/results.md`.

## Diagnostics

Two tools with no AlphaZero equivalent, because AlphaZero has no learned
model that could be wrong:

```sh
./diag_eval <checkpoint-path> [simulations]
# default: 150 simulations. Plays one game as X and one as O against
# minimax, printing every move and the MCTS visit distribution behind it —
# useful for tracing exactly where and why a checkpoint loses.

./latent_probe <checkpoint-path> [trajectories]
# default: 500 trajectories. Rolls the dynamics network forward along
# real games and compares its imagined latents against a fresh
# representation pass on the real board those actions reach.
```

Sample `latent_probe` output (200 trajectories, against one of the
project's checkpoints):

```
Learned model vs. real game, over 200 trajectories
Imagined latent (dynamics rolled k steps) compared against a fresh
representation pass on the real board the same actions reach.

  k   samples   mean |value error|   mean policy distance
  1       200              0.2052                  0.1399
  2       196              0.2377                  0.2837
  3       182              0.2970                  0.4040
  4       166              0.3159                  0.4859
  5       145              0.3926                  0.5453
  6        98              0.4426                  0.6262

Terminal reward prediction: mean |error| = 0.8160 over 102 transitions
```

Read it as: `mean policy distance` is the total-variation distance between
what the network's prediction head believes about an *imagined* position at
depth `k` and what it says about the *real* board those same moves reach —
0 means the model agrees with reality, 1 means it is completely wrong. It
climbs steadily with depth, which is the dynamics model drifting from the
real game the further it is rolled forward without ever being corrected by
a real observation.

## Project layout

```
include/mz/   public headers for each component
src/          implementations (board, minimax, network, mcts, targets,
              replay buffer, self-play, minimax-eval helper)
apps/         the three executables (train, evaluate, play_cli)
tools/        diag_eval and latent_probe (diagnostics); render_doc.py (the
              hand-run, offline markdown-to-HTML converter for docs/); and
              check_citations.py (verifies the file:line citations in
              docs/algorithm-explained.md against the actual source)
tests/        assert-based test executables (one per component) plus
              tests/integration_smoke.sh, an end-to-end pipeline check
docs/         the algorithm walkthrough (markdown and illustrated HTML),
              the MuZero-vs-AlphaZero comparison, and the training results
```

## Deliberately out of scope

- **MuZero Reanalyze** — periodically re-running search over stored
  trajectories with the current network and overwriting their targets in
  place. Not implemented, but the replay buffer stores whole trajectories
  and target construction is a separate, pure, re-runnable step
  specifically so it can be added later without restructuring. See the
  spec's "Reanalyze Readiness" section.
- **The categorical value/reward representation from the paper** (a
  softmax over a 601-bin support with an invertible scaling transform).
  Scalar `tanh` heads are used instead, since tic-tac-toe's rewards are
  exactly `{-1, 0, +1}`. See the spec's "Deviations From the Paper"
  section.
- **The AlphaZero arena/network-promotion step**, matching the sibling
  project. See the spec's "Non-Goals" section.

## Authorship

Mark Castelluccio, designed and implemented with [Claude Code](https://claude.com/claude-code).
