# Training results

## Headline result

Across five training runs (400–800 iterations each), MuZero self-play
against a learned model **did not reach `losses=0` and hold it**. The best
checkpoints found (from two independent runs, `numSimulations=100`)
score:

```
$ ./build/evaluate checkpoint.bin 50
vs minimax over 50 games/side: wins=0 draws=50 losses=50
```

`wins` was 0 in every single evaluation across every run (100+ evaluation
calls total) — the minimax opponent and evaluation harness are correct.
What is not fixed is `losses`: the best checkpoints reliably solve one
side of the board (X or O, depending on the run) and reliably lose a
specific line on the other side. `diag_eval` (below) shows the exact ply.
This is reported honestly, per the task brief, as the best result reached
within the allotted training-run budget — not a fabricated pass.

## Final evaluation

Games per side: 50 (100 games total). Best checkpoints (two separate
400-iteration runs converged to this same split):

| checkpoint | wins | draws | losses |
|---|---|---|---|
| `checkpoint_run2.bin` (`numSimulations=100`, defaults otherwise) | 0 | 50 | 50 |
| `checkpoint_run4.bin` (`numSimulations=100`, `learningRate=0.01`) | 0 | 50 | 50 |

Both checkpoints show the same shape at the smaller 20-games/side
evaluation used during training (`wins=0 draws=20 losses=20`), i.e. the
result is reproducible, not a one-off sampling artifact — see "Why the
losses land in round numbers" below.

The other three runs (baseline `numSimulations=60`, `numSimulations=160`,
and a second `numSimulations=100`/default-learning-rate run at 800
iterations with a different random seed) ended at `wins=0 draws=0
losses=100` — complete failure, despite one of them (the 800-iteration
run) using twice the training of the successful ones. See "Tuning
history" for the full detail; the seed-to-seed variance turned out to
matter more than any hyperparameter we changed.

## Why the losses land in round numbers (0/20/40, 0/50/100)

`evaluateAgainstMinimax` (`src/eval.cpp`) plays with a **fixed RNG seed**,
**no root Dirichlet noise**, and minimax opponents that are themselves
deterministic. Every one of the `gamesPerSide` games played as X is
therefore the identical game, and likewise for O. The eval score is
consequently always an exact multiple of `gamesPerSide`: a run either
solves a side completely or fails it completely, with nothing in between.
This is expected behavior of the harness (not a bug), but it does mean a
single tactical blind spot on one side is enough to pin `losses` at
exactly half of all games, and it explains why intermediate scores in the
logs jump between 0, 20, and 40 (or 0, 50, 100) rather than varying
smoothly.

## Diagnosing the loss: `diag_eval`

Both `checkpoint_run2.bin` and `checkpoint_run4.bin` fail at the same
*kind* of ply: an immediate three-in-a-row block. `checkpoint_run2.bin`
loses playing O:

```
=== network plays O ===
    . . .          X . .          X X .
    . . .    ->    . O . -- minimax plays 1 -->  . O .
    . . .          . . .          . . .
  ply 3  network -> 8   rootValue=-0.203  depth=11
  visits:  2:0.11  3:0.21  5:0.00  6:0.07  7:0.03  8:0.58
    X X .
    . O .
    . . O
  ply 4  minimax -> 2   (X X X top row)
  result: network LOST
```

After X plays corner-then-adjacent-edge (0, then 1), X threatens an
immediate win at square 2. The network's own MCTS visit distribution
puts only 11% of its visits on the one square that blocks it (2), and
58% on an unrelated corner (8). `checkpoint_run4.bin` shows the mirror
image: it loses as X to the same class of missed block (a column threat
after O plays 0 and 6, needing a block at square 3), with visits
`1:0.29  3:0.21  5:0.18  7:0.11  8:0.21` — the correct block (3) gets
only 21% of visits, versus 29% spent on an unrelated corner.

In both cases the network is not confidently wrong — it is *unconcentrated*:
search spreads real probability mass over several non-blocking moves
instead of collapsing onto the one legal, correct one. That is consistent
with the project's expected result: part of the simulation budget is
spent on dynamics-network rollouts into actions the real game would
never allow, so less budget is left to firmly resolve the one square that
matters. The AlphaZero sibling project's README describes the identical
failure shape ("confidently losing every game from one side because
search never visits the one move that mattered enough") even though its
search steps real rules — but MuZero's added source of imprecision (the
learned dynamics, see the `latent_probe` section below) is on top of it,
and reaching zero losses took AlphaZero 20 iterations against MuZero's
400+ without doing so.

## Iterations to `losses=0`

- **First reached 0** (training-time eval, 20 games/side): iteration 390
  of run 4 (`numSimulations=100`, `learningRate=0.01`) —
  `eval vs minimax: wins=0 draws=40 losses=0`. This was transient: the
  very next training-time eval (iteration 400, the run's last) had
  regressed to `draws=20 losses=20`, and because checkpoints are
  overwritten every 10 iterations, the momentary 0-loss network itself
  was not preserved for separate re-evaluation.
- **Sustained at 0**: not achieved in any of the five runs within budget.
  The closest sustained state is `losses=20` (out of 40 training-time
  eval games), held for the last 100+ iterations of run 2
  (`numSimulations=100`, default learning rate) — 11 consecutive
  evaluation checkpoints (iterations 300–400) all read
  `wins=0 draws=20 losses=20`.

## Comparison with the AlphaZero sibling project

From `~/projects/alphazero-tictactoe/README.md`: with root Dirichlet
noise during self-play, a real run reached `draws=40 losses=0` (all 40
games, both sides) after **20 iterations**.

MuZero here did not reach that bar within 400–800 iterations. The
comparison is not apples-to-apples, though, and the gap should not be
read as a precise multiplier. What differs between one "iteration" in
each project (read directly from each project's `apps/train.cpp`):

| | AlphaZero | MuZero (this project, tuned) |
|---|---|---|
| Self-play games/iteration | 25 | 25 |
| MCTS simulations/move (self-play) | 50, over the **real** rules | 100 (tuned up from 60), over a **learned** dynamics model |
| Gradient steps/iteration | 20 | 40 |
| Batch size | 32 | 64 |
| Samples/iteration (steps × batch) | 640 | 2,560 |
| Learning rate | 0.01 | 0.02 (0.01 also tried, see below) |
| Replay buffer capacity | 10,000 positions | 2,000 games |

So each MuZero iteration does 4x the gradient-step throughput of an
AlphaZero iteration (40×64 vs 20×32), and even so needed 20x+ as many
iterations to approach (not reach) the same bar — a rough 80x+ more total
gradient-step-samples processed, for a materially worse final result.
That gap, not any single number in it, is the finding the spec predicted:
MuZero pays for not being told the rules, both in wasted search budget
(the `diag_eval` transcripts above) and in a dynamics model that itself
drifts from the real game (the `latent_probe` table below). The
per-iteration figures above should be read as context for that gap, not
as inputs to a precise "N times slower" claim — the two self-play
loops, network architectures, and target constructions differ in enough
other ways (K-step unrolling, a reward head, a representation network)
that iteration count is not a unit-comparable currency between them.

## `latent_probe` table

Run against `checkpoint_run2.bin` (the representative final checkpoint,
matching the hyperparameters left committed in `apps/train.cpp`), over
2000 sampled trajectories:

```
  k   samples   mean |value error|   mean policy distance
  1      2000              0.1948                  0.1820
  2      1952              0.2314                  0.2601
  3      1840              0.2864                  0.3286
  4      1606              0.3161                  0.3847
  5      1296              0.3541                  0.4412
  6       847              0.3774                  0.4822

Terminal reward prediction: mean |error| = 0.8426 over 1153 transitions
```

(`checkpoint_run4.bin` shows the same shape: value error 0.29→0.29,
policy distance 0.20→0.36 from k=1 to k=6, terminal reward error 0.79.)

**Reading the drift**: both error columns climb steadily and roughly
monotonically as the dynamics network is rolled further from a real
board — by k=6 the imagined latent disagrees with a fresh representation
pass on the real board almost half the time in policy terms (0.48 total
variation, versus 0.18 at k=1). The policy-distance column is the more
informative one here (as the brief warned): it tracks real drift from
k=1 onward, while the value-error column starts smaller-but-nonzero and
grows more slowly in relative terms because the value head is fairly flat
across boards this early in training, which mutes its apparent error.
Values in the game are tightly bounded (win/draw/loss), so even a poorly
differentiated value head cannot be off by much in absolute terms — the
policy distribution, over 9 possible moves including illegal ones the
model has never been corrected on, has much more room to diverge, and
does.

The terminal-reward error (0.84, on rewards that are themselves in
`{-1, 0, +1}`) is the most damning single number: the dynamics network is
not reliably predicting when the game has ended or who won when it
imagines forward. A search that cannot trust its own model's sense of
"the game is over and X just won" cannot correctly value the branches
below the point where it stops trusting real transitions — which lines
up exactly with the diagnosed failure above, where search spreads its
budget across squares instead of concentrating on the one forced reply.

## Tuning history

All runs from a freshly-initialized network, `gamesPerIteration=25`,
`batchSize=64`, `trainStepsPerIteration=40`, `bufferCapacity=2000` games,
`evalIntervalIterations=10`, `evalGamesPerSide=20`, `evalSimulations=150`,
`unrollSteps=5`, `tdSteps=32` (all as committed at the start of this
task) unless noted. Wall-clock times measured with `time` around
`./build/train`.

| run | iterations | changed from defaults | wall-clock | training-time eval trend | final `./build/evaluate ... 50` |
|---|---|---|---|---|---|
| 1 | 400 | none (baseline: `numSimulations=60`, `learningRate=0.02`) | 44.4s | `losses=40` for all but one blip (draws=20 at iter 320, reverted next eval); ends `losses=40` | `draws=0 losses=100` |
| 2 | 400 | `numSimulations=100` | 49.3s | `losses=40` until iter ~250, then `draws=20 losses=20` sustained for the last 11 consecutive evals (iters 300–400) | `draws=50 losses=50` |
| 3 | 400 | `numSimulations=160` | 57.7s | worse than run 2: `losses=40` in 36/40 evals, only 4 partial (`draws=20`) evals, none sustained; ends `losses=40` | `draws=0 losses=100` |
| 4 | 400 | `numSimulations=100`, `learningRate=0.01` (halved) | 49.2s | noisier than run 2: 31/40 `losses=40`, 8/40 `draws=20`, one single eval at iter 390 hit `draws=40 losses=0`, then regressed to `draws=20 losses=20` by iter 400 | `draws=50 losses=50` |
| 5 | 800 | same config as run 2 (`numSimulations=100`, default `learningRate=0.02`), different random seed, double the iterations | 99.5s | `losses=40` on **every one of 80 evals**, no improvement at all despite 2x the training of run 2 | `draws=0 losses=100` |

Reading across runs 2 and 5 — identical hyperparameters, only the random
seed and iteration count differ — is the single most important result of
this tuning pass: **seed-to-seed variance dominated every hyperparameter
change we tried.** Bumping `numSimulations` from 60 to 100 (run 1 → run
2) was the only change that produced a repeatable partial improvement
(runs 2 and 4 both landed on the identical `draws=50 losses=50` final
score from different learning rates); pushing it further to 160 (run 3)
made things worse, not better, matching the brief's caution that more
search is not simply always better once real signal is scarce. Halving
the learning rate (run 4) changed *which* iterations looked good but not
the final outcome. Doubling training length with the same seed-sensitive
setup (run 5) was actively worse than the shorter run 2, which is the
clearest evidence that what we're tuning against here is largely noise at
this iteration budget, not a smooth function of the hyperparameters.

`trainStepsPerIteration`/`batchSize` (tuning step 3) and `unrollSteps`
(step 4) were not tried — the training-run budget for this task (3-5
runs) was spent establishing that `numSimulations` and `learningRate`
were not the deciding factor, which felt like the more load-bearing
finding to nail down with the runs available. See "What to try next".

### `tdSteps` (bootstrapping)

Not tried. `tdSteps` was next in the brief's tuning order after
`unrollSteps`, and the training-run budget was already spent (five runs)
establishing the seed-variance finding above. This is a real gap in the
tuning pass, not a decision that bootstrapping wouldn't matter — see
"What to try next".

## Final hyperparameters

Left committed in `apps/train.cpp`:

| parameter | value | changed? | why |
|---|---|---|---|
| `numSimulations` (`SelfPlayConfig`) | 100 | yes, from 60 | Run 1 (60, the committed starting point) never posted a single non-`losses=40` training-time eval outside one blip. Run 2 (100) was the first change tried, per the brief's tuning order, and was the only change that produced a repeatable partial result across two independent runs (2 and 4). Run 3 showed 160 is not simply better — see tuning history. |
| `learningRate` | 0.02 | no (left at the starting-point default) | Run 4 tried halving it to 0.01 and got the same final score (`draws=50 losses=50`) as run 2's unchanged 0.02, just with a noisier path there; run 5 shows the same 0.02 config can also fail completely on a different seed. Since neither value demonstrated a repeatable advantage, the default was kept to minimize the number of simultaneous changes against the committed starting point. |
| `gamesPerIteration` | 25 | no | not tuned this pass |
| `batchSize` | 64 | no | not tuned this pass |
| `trainStepsPerIteration` | 40 | no | not tuned this pass |
| `unrollSteps` (`TargetConfig`) | 5 | no | not tuned this pass |
| `tdSteps` (`TargetConfig`) | 32 (bootstrap term falls off the end; value target = game outcome) | no | not tuned this pass, see above |
| `bufferCapacity` | 2000 games | no | starting-point default |
| `evalIntervalIterations` / `evalGamesPerSide` / `evalSimulations` | 10 / 20 / 150 | no | starting-point defaults |

Running `./build/train 400 checkpoint.bin` with the code as committed
reproduces the setup used for runs 2 and 5 (only the random seed differs
between invocations, since the network and RNG are seeded from
`std::random_device` on each run) — meaning a fresh run of the committed
code has, on this evidence, roughly even odds of landing on run 2's
partial success or run 5's complete failure.

## Wall-clock time and machine

- Machine: `Apple M4` (`sysctl -n machdep.cpu.brand_string`), 10 cores
  (`sysctl -n hw.ncpu`).
- Run times (400 iterations unless noted), single-threaded `train`
  binary, Release build: 44.4s (run 1) / 49.3s (run 2) / 57.7s (run 3) /
  49.2s (run 4) / 99.5s (run 5, 800 iterations). All five runs combined:
  under five minutes of wall-clock time — far under the "tens of
  minutes" budget anticipated for a single 400-iteration run, since this
  problem's self-play games are extremely short (at most 9 plies) and
  the network is small.

## What to try next

In priority order, given what runs 1–5 actually showed:

1. **`tdSteps`.** Untried this pass, and explicitly called out in the
   brief as worth a paragraph either way. Turning on real bootstrapping
   (e.g. `tdSteps=5`) changes what the value target rewards early in a
   game, before the outcome is known, which is exactly where our
   `diag_eval` transcripts show search failing to commit to forced
   moves — a better early-game value signal seems more likely to fix a
   4-ply tactical blind spot than a search-budget or learning-rate change
   did.
2. **Averaging over seeds rather than over hyperparameters.** Given how
   large the run-to-run variance was at fixed hyperparameters (run 2 vs.
   run 5), the next tuning pass should hold a hyperparameter fixed and
   run 3+ seeds before concluding anything about it, rather than treating
   a single run's outcome as attributable to the change under test — this
   pass's run 3 (`numSimulations=160`, one seed) may simply have drawn an
   unlucky seed rather than genuinely being worse than 100.
3. **`trainStepsPerIteration`/`batchSize`** (brief's tuning step 3,
   untried): more gradient steps per unit of self-play, to see whether
   the network is under-trained relative to the data it already collects
   rather than under-explored.
4. Longer runs with root-Dirichlet-noise diagnostics: confirm whether the
   specific missed-block pattern in `diag_eval` recurs at higher
   iteration counts, or whether it is a different specific line each
   time (which would point toward search-budget rather than a
   consistently mis-learned policy region).

## Global constraints observed

No changes were made to `src/mcts.cpp`, `src/network.cpp`,
`src/targets.cpp`, or `src/selfplay.cpp`. Legality was never masked below
the search root at any point in this tuning pass — search below the root
in every run above was free to (and did) wander into illegal branches,
per the project's design.
