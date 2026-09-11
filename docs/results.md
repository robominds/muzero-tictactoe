# Training results

## Bottom line, stated plainly

**The spec's first success criterion — converging to drawing play against
minimax — was not met.** Across 23 training runs (5 initial single-seed
runs plus two 3-seed-per-cell grids, 400 to 1500 iterations each), the
best configuration found (`numSimulations=100`, `temperatureMoves=6`,
800 iterations) converged on **one of three seeds tested**. The other two
seeds at that exact configuration ended at `draws=50 losses=50` and
`draws=0 losses=100` respectively. A configuration that converges on one
seed out of three has not converged, and no configuration tried converged
on all three seeds of any cell.

**The spec's second success criterion — that `latent_probe` show rolling
dynamics staying close to a fresh representation pass — was also not
met.** The measurements are below, in "`latent_probe`: even the converged
checkpoint's model still drifts": policy distance rises from 0.24 at k=1
to 0.55 at k=6, value error sits around 0.31–0.34, and terminal-reward
error is 0.77 on rewards in `{-1,0,+1}`. Neither of the spec's two success
criteria was met.

The one success is real and is reported below in full, because it is the
only direct evidence in this project of what convergence looks like and
because it pins down exactly what a converged checkpoint's search and
policy look like in contrast to a failing one. But it is one lucky seed,
not a solved configuration, and this document does not present it as
more than that.

```
$ ./build/evaluate ckpt_sim100_it800_seed303.bin 50
vs minimax over 50 games/side: wins=0 draws=100 losses=0
```

`wins` was 0 in every evaluation across every one of the 23 runs (over
150 separate `evaluate`/training-time-eval calls) — the minimax opponent
and the evaluation harness are correct throughout. What did not
reliably resolve is `losses`.

A `draws=50 losses=50` result — the second most common outcome across
every grid — is not "half the games are bad." `evaluateAgainstMinimax`
(`src/eval.cpp`) plays with a fixed RNG seed, no root Dirichlet noise, and
a deterministic minimax opponent, so every game played as X is the
identical game, and likewise for O. A `draws=50 losses=50` split means
**the network plays one side perfectly and walks into the exact same
losing line as the other side, every single time it is asked.** It is one
wrong move in one position, repeated 50 times, not diffuse weakness. The
`diag_eval` transcripts below show exactly which move and where.

## What actually localizes the failure

A direct probe of two 1500-iteration checkpoints, querying
`initialInference` on hand-built positions, separates the value head from
the policy head:

- **Value head: correct.** On a position where the player to move has an
  immediate win, the value head reports roughly +0.54 to +0.65. On a
  position where the player to move is in trouble, it reports roughly
  −0.41 to −0.66. Right signs, sensible magnitudes, in both directions.
  If the value target or the training gradient had a sign error, this is
  the first thing it would destroy — it did not.
- **Policy head: wrong in exactly the way that matters.** On "X to move,
  and square 2 wins outright," the policy head's argmax lands on square 5
  or 8, with only 0.09 to 0.32 of its mass on the winning move. It does
  not see an immediate win.
- **Search does not rescue the policy head, and cannot.** `diag_eval` at
  400 simulations reports search tree depths of **18–19 on a nine-square
  board** (we independently reproduced 18; a second measurement reported
  19). A tic-tac-toe game cannot be more than 9 plies deep. Depth 18 means
  most of a 400-simulation budget is spent below the point where any
  legal game could still be running — imagined moves layered on imagined
  moves, in positions the dynamics network invented and the real rules
  would never reach. This is root-only legality doing exactly what the
  spec predicted, more expensively than expected: it is not merely
  "search sometimes explores an illegal branch," it is "the median
  simulation is illegal by the time it terminates."

This rules out a sign bug or a broken training target as the cause — the
gradient is verified per-parameter across all eight layers, the search
backup was hand-derived, and the n-step targets were hand-computed
against a fixture in earlier tasks; the value head's correct sign
structure in both directions is exactly the evidence a sign error would
not survive. The failure is localized to the policy head not learning a
rare tactical pattern reliably, compounded by search that cannot fall
back on ever touching real rules below the root to correct it.

## `diag_eval`: the losing line, twice, in two different configurations

**`numSimulations=100`, seed 202 (`draws=50 losses=50`), losing as O, at
the trained simulation count:**

```
=== network plays O ===
ply 1  network -> 5   visits: 1:0.03 2:0.06 3:0.03 4:0.13 5:0.39 6:0.07 7:0.02 8:0.27
ply 2  minimax -> 2        (board: X . X / . . O / . . .)
ply 3  network -> 8   visits: 1:0.04 3:0.08 4:0.15 6:0.09 7:0.19 8:0.45
ply 4  minimax -> 1        (X X X top row) -- network LOST
```

X has corners 0 and 2 after ply 2, threatening an immediate win at square
1. The network's own MCTS visit distribution puts only **4%** of its
visits on the one square that blocks it (1), and **45%** on an unrelated
corner (8).

**`numSimulations=300`, seed 101 (`draws=0 losses=100`), losing as X, at
the trained simulation count:**

```
=== network plays X ===
ply 0  network -> 4
ply 1  minimax -> 0
ply 2  network -> 5   visits: 1:0.00 2:0.09 3:0.06 5:0.83 6:0.02 7:0.00 8:0.00
ply 3  minimax -> 3        (board: O . . / O X X / . . .)
ply 4  network -> 2   visits: 1:0.00 2:0.47 6:0.41 7:0.02 8:0.10
ply 5  minimax -> 6        (O X . / O X X / O . .  -- left column) -- network LOST
```

Same shape: O has 0 and 3 (left column) after ply 3, threatening a win at
square 6. The network's visits split 47% on square 2 (which blocks
nothing) versus 41% on square 6 (the actual, only correct block) — close
enough this time that it reads as genuine indecision rather than the
lopsided 4%-vs-45% miss on the O side, but the network still plays the
higher-visit, non-blocking move and loses. Two different configurations
(100 vs. 300 simulations, two different seeds), and both fail at the
identical *kind* of ply: an immediate three-in-a-row the network must
block, with search spread across multiple candidates instead of
collapsed onto the one legal correct one.

For contrast, the one converged checkpoint (`numSimulations=100`, seed
303, probed here at `diag_eval`'s default 150 simulations — the same
150 that `./build/evaluate` uses regardless of a checkpoint's trained
simulation count, so this is the same search budget behind its
`draws=100 losses=0` score above) handles the same class of position
correctly — at the analogous ply in its own transcript it puts 74–80% of
its visits on the correct block and plays it:

```
=== network plays X (converged checkpoint) ===
ply 2  network -> 2   visits: 1:0.11  2:0.74  3:0.03  5:0.03  6:0.00  7:0.01  8:0.09
=== network plays O (converged checkpoint) ===
ply 3  network -> 2   visits: 2:0.80  3:0.04  5:0.03  6:0.00  7:0.00  8:0.13
```
Both of its games (as X and as O) end in `draw`.

## The two grids run this pass

Everything held at `gamesPerIteration=25`, `batchSize=64`,
`trainStepsPerIteration=40`, `learningRate=0.02`, `bufferCapacity=2000`
games, `unrollSteps=5`, `tdSteps=32`, unless the column says otherwise.
**Each run's random seed now controls everything** — network weight
initialization and all self-play/training randomness — via a new
optional third CLI argument to `apps/train.cpp` (see "Reproducibility"
below), so every cell below is exactly reproducible.

### Grid 1 — `temperatureMoves` × `numIterations` (`numSimulations=100` held fixed)

| temperatureMoves | iterations | seed 101 | seed 202 | seed 303 | converged / 3 |
|---|---|---|---|---|---|
| 2 (baseline) | 400 | `draws=50 losses=50` | `draws=0 losses=100` | `draws=0 losses=100` | 0/3 |
| 2 (baseline) | 1500 | `draws=0 losses=100` | `draws=0 losses=100` | `draws=0 losses=100` | 0/3 |
| 6 | 400 | `draws=0 losses=100` | `draws=0 losses=100` | `draws=0 losses=100` | 0/3 |
| 6 | 1500 | `draws=50 losses=50` | `draws=0 losses=100` | `draws=0 losses=100` | 0/3 |

(Scores are `./build/evaluate <checkpoint> 50` on each cell's final
checkpoint.) Neither `temperatureMoves` value converged on any seed at
either iteration count. `temperatureMoves=6, iterations=1500, seed=303`
is worth a specific mention even though its `evaluate` score is a flat
`losses=100`: its **training-time** eval line (20 games/side, every 10
iterations) spent a long stretch — roughly iterations 640–950 — mostly at
`losses=0` or `losses=20`, closer to sustained convergence than anything
else in this grid, before drifting back to `losses=40` by iteration 1500.
That the network can pass through a long near-converged stretch and drift
back out of it, with no early-stopping or best-checkpoint tracking in
`apps/train.cpp`, is itself informative: whatever `apps/train.cpp` saves
as "final" is whatever the last 10-iteration window happened to produce,
not necessarily the best point the run reached.

The plain reading of Grid 1: at `numSimulations=100`, doubling or
tripling `temperatureMoves` did not produce a repeatable improvement
distinguishable from seed noise, at either 400 or 1500 iterations, and
1500 iterations was not obviously better than 400 — the training loss
was still falling at the end of every run (never plateaued; see
per-run loss traces in the raw logs), but the categorical eval score
did not track it: `temperatureMoves=2, seed=303` was `losses=100` at both
400 *and* 1500 iterations, unchanged despite 3.75x more training.

### Grid 2 — `numSimulations` × 3 seeds (`temperatureMoves=6`, 800 iterations held fixed)

| numSimulations | seed 101 | seed 202 | seed 303 | converged / 3 |
|---|---|---|---|---|
| 100 | `draws=0 losses=100` | `draws=50 losses=50` | **`draws=100 losses=0`** | **1/3** |
| 300 | `draws=0 losses=100` | `draws=0 losses=100` | `draws=0 losses=100` | 0/3 |

This is the cleanest single result in the whole tuning pass, because it
is a matched-seed comparison (same three seeds, same everything else):
**tripling the search budget from 100 to 300 simulations did not help —
it went 1/3 converged down to 0/3.** This directly contradicts the
hypothesis that more search budget was the untried lever. The
search-depth finding above explains why: at 100 simulations the tree
already reaches depth 15–18 on a 9-ply-max game; giving it 300 does not
make the *legal* portion of the tree deeper or better-resolved, it mostly
buys more simulations spent even further past the point of any legal
game, over a dynamics model whose own drift is unbounded with depth (see
`latent_probe` below). Search budget was not the untried lever that
mattered; if anything this grid is evidence the branching-factor argument
(more sims needed because effective branching never shrinks) is
outweighed in practice by how unreliable the *deep* part of the tree is
regardless of budget.

Caveat on the one success: this grid did not include a
`temperatureMoves=2, numSimulations=100`, seed-303, 800-iteration cell,
so it is not established whether `temperatureMoves=6` specifically caused
seed 303 to converge, or whether seed 303 would have converged at
`temperatureMoves=2` too. That specific ablation — same seed, same
`numSimulations=100`, same 800 iterations, `temperatureMoves=2` instead
of 6 — is the single most informative next run to isolate the effect,
and was not run within this pass's budget.

## Reproducibility added this pass

`apps/train.cpp` previously seeded everything (`MuZeroNetwork`'s initial
weights, the self-play/training RNG, and — undocumented until this pass —
`ReplayBuffer`'s internal sampling RNG) from `std::random_device`, making
every run's outcome unrepeatable. To run the seed grids above honestly,
three changes were made:

- `apps/train.cpp` takes an optional third argument, seed, defaulting to
  `std::random_device{}()` when omitted (old behavior preserved). The
  seed now drives `MuZeroNetwork`'s weight initialization (via its
  existing `MuZeroNetwork(std::uint32_t seed)` constructor — no change
  needed there), the self-play/training `std::mt19937`, and the replay
  buffer's sampling RNG (see next point) — one number reproduces an
  entire run bit-for-bit, verified by `cmp`-ing two checkpoints trained
  from the same seed.
- `include/mz/replay_buffer.hpp` / `src/replay_buffer.cpp`: `ReplayBuffer`
  had a private `std::mt19937 rng_` seeded from `std::random_device` with
  no way to control it. Added an optional `seed` constructor parameter
  (default `std::random_device{}()`, so every other caller and every
  existing test is unaffected) and threaded it through to the member
  initializer. This file is not on the forbidden list (`mcts.cpp`,
  `network.cpp`, `targets.cpp`, `selfplay.cpp`) — it is a data structure,
  not algorithm code, and no sampling *behavior* changed, only where its
  RNG's seed comes from.

All 10 existing tests pass unmodified after both changes. This is a
real, load-bearing methodological gap that was open for the entire task
until now: every run reported before this pass (runs 1–5, and this
task's first attempt at a `temperatureMoves` grid before the coordinator
caught it) used an uncontrolled seed and could not be reproduced or
matched across configurations.

## `latent_probe`: even the converged checkpoint's model still drifts

Run against the one converged checkpoint (`numSimulations=100`,
`temperatureMoves=6`, seed 303, 800 iterations), over 2000 trajectories:

```
  k   samples   mean |value error|   mean policy distance
  1      2000              0.3128                  0.2422
  2      1952              0.2316                  0.4124
  3      1840              0.3268                  0.4499
  4      1606              0.2988                  0.4725
  5      1296              0.3371                  0.5042
  6       847              0.3305                  0.5501

Terminal reward prediction: mean |error| = 0.7682 over 1153 transitions
```

The most important reading of this table is the *contrast* with the fact
that this exact checkpoint plays perfectly against minimax (`diag_eval`
above, `draws=100 losses=0`). **A converged, perfectly-drawing checkpoint
still has a dynamics model whose imagined policy disagrees with the real
game over half the time by depth 6 (0.55 total variation), and whose
terminal-reward sense is off by 0.77 on rewards in `{-1,0,+1}`.** Model
fidelity at depth is not what determined whether this run converged —
what determined it was whether the root-level policy and value were
correct enough, and whether self-play's exploration (governed largely by
`temperatureMoves` and pure seed luck) ever produced and reinforced the
training examples that correct the policy head's blind spots near the
root. The deep dynamics model can stay wrong; convergence in this project
is a property of the shallow, frequently-visited part of the tree, not of
the model as a whole. This reframes the search-depth finding above:
search wasting budget past depth 9 is wasteful, but it is not obviously
*harmful* to a converged network the way it might be to an unconverged
one, because a converged network's root policy is already good enough
that deep, wrong simulations get outvoted by shallow, right ones. An
unconverged network has no such shallow anchor, so the same deep noise
has nothing to be outvoted by.

## Full tuning history (all 23 training runs across this task)

**Phase 1 — single seed, uncontrolled (`std::random_device`), 400–800 iterations:**

| run | iterations | changed from prior | wall-clock (solo) | outcome (`evaluate ... 50`) |
|---|---|---|---|---|
| 1 | 400 | none (`numSimulations=60`, baseline) | 44.4s | `draws=0 losses=100` |
| 2 | 400 | `numSimulations=100` | 49.3s | `draws=50 losses=50` |
| 3 | 400 | `numSimulations=160` | 57.7s | `draws=0 losses=100` |
| 4 | 400 | `numSimulations=100`, `learningRate=0.01` | 49.2s | `draws=50 losses=50` |
| 5 | 800 | same as run 2, different (uncontrolled) seed | 99.5s | `draws=0 losses=100` |

**Phase 2 — Grid 1, `temperatureMoves` × `numIterations`, 3 controlled seeds each (`numSimulations=100`):**
See the Grid 1 table above. 0/12 cells converged. Batch wall-clock
(6-way parallel): ~3.6 min for the `temperatureMoves=2` batch, ~3.8 min
for the `temperatureMoves=6` batch (contended; not clean per-run
figures — see solo timings in Phase 1 for the uncontended cost of a
400-iteration run).

**Phase 3 — Grid 2, `numSimulations` × 3 controlled seeds (`temperatureMoves=6`, 800 iterations):**
See the Grid 2 table above. 1/6 cells converged (`numSimulations=100`,
seed 303). Batch wall-clock (6-way parallel, `numSimulations=100` and
`=300` launched together): ~3 min for the `numSimulations=100` trio to
finish, ~4 min total for both trios (the `=300` trio runs roughly 2x
longer per-iteration than `=100`, consistent with search cost scaling
with simulation count).

Total: 23 training runs, 1 convergence (`draws=100 losses=0`), 5 partial
results (`draws=50 losses=50`: Phase 1 runs 2 and 4; Grid 1's
`temperatureMoves=2/400/seed101` and `temperatureMoves=6/1500/seed101`;
Grid 2's `numSimulations=100/seed202`), 17 complete failures
(`draws=0 losses=100`). Every single one of the 23 runs had `wins=0`.

## Comparison with the AlphaZero sibling project

From `~/projects/alphazero-tictactoe/README.md` and its
`apps/train.cpp`: with root Dirichlet noise during self-play, a real run
reached `draws=40 losses=0` (all 40 games, both sides) after **20
iterations**. MuZero here converged on 1 of 3 seeds after 800 iterations
at its best-found configuration, and 0 of 3 seeds at every other
configuration tried, up to 1500 iterations.

That AlphaZero number is a single reported run, with no seed control and
no grid — the sibling project has no multi-seed evidence at all, unlike
the matched 3-seed-per-cell sweep this document runs for MuZero. The two
results below are not measured to the same standard, and the comparison
should be read with that asymmetry in mind.

What differs between one "iteration" in each project (read directly from
each project's `apps/train.cpp`):

| | AlphaZero | MuZero (this project, best found) |
|---|---|---|
| Self-play games/iteration | 25 | 25 |
| MCTS simulations/move (self-play) | 50, over the **real** rules | 100, over a **learned** dynamics model |
| Gradient steps/iteration | 20 | 40 |
| Batch size | 32 | 64 |
| Samples/iteration (steps × batch) | 640 | 2,560 |
| Learning rate | 0.01 | 0.02 |
| Replay buffer capacity | 10,000 positions | 2,000 games |

Each MuZero iteration does 4x the gradient-step throughput of an
AlphaZero iteration, and even so, 800 iterations (worth roughly 80x
AlphaZero's total gradient-step-samples) converged on only one of three
seeds, where AlphaZero's much smaller compute budget converged in 20, on
the single run reported in its README. The per-iteration figures above
are context, not a precise multiplier — the two self-play loops, network
architectures (a representation network and a reward head that AlphaZero
has no equivalent of), and target constructions (K-step unrolling) differ
enough that iteration count is not a unit-comparable currency between
the projects. The comparison that *is* precise: AlphaZero's search never
leaves the real board, so its greedy self-play tail reliably rediscovers
any corrective line; MuZero's search increasingly imagines positions the
real game cannot reach (depth 18 on a 9-ply game), and its greedy
self-play tail can lock onto a wrong belief indefinitely because it never
revisits the position that would correct it. That is the cost of not
being given the rules, and this domain is small enough to make the
contrast unambiguous even though the exact iteration multiplier is not a
clean number.

## Final hyperparameters left committed

| parameter | value | changed? | why |
|---|---|---|---|
| `numSimulations` (`SelfPlayConfig`, in `include/mz/selfplay.hpp`) | 100 | yes, from 60 | Grid 2 is a clean, matched-seed comparison showing 300 converged on 0/3 seeds versus 100's 1/3 — more search budget was not simply better, consistent with the search-depth finding (budget is mostly spent past the point where any legal game could still be running). 100 is an empirical middle ground, not a principled optimum; 160 (Phase 1, single seed) also did no better than 100. |
| `temperatureMoves` (`SelfPlayConfig`, in `include/mz/selfplay.hpp`) | 6 | yes, from 2 | The only sustained `draws=100 losses=0` result in the entire task used this value. However, Grid 1 (a direct, matched-seed comparison of `temperatureMoves=2` vs. `6` at `numSimulations=100`) found 0/12 cells converged at *either* value, so this parameter's effect is not conclusively established by the evidence collected — see "What to try next" for the specific ablation that would settle it. Kept at 6 on the strength of (a) the one success using it and (b) the mechanistic argument in the header comment (MuZero's greedy self-play tail searches a drifting model and cannot self-correct the way AlphaZero's real-rules tail can), not because Grid 1 proved it. |
| `learningRate` | 0.02 | no | Phase 1 run 4 tried 0.01 and reached the same final score as run 2's 0.02; no repeatable advantage either way. |
| `gamesPerIteration` | 25 | no | not tuned this pass |
| `batchSize` | 64 | no | not tuned this pass |
| `trainStepsPerIteration` | 40 | no | not tuned this pass |
| `unrollSteps` (`TargetConfig`) | 5 | no | not tuned this pass |
| `tdSteps` (`TargetConfig`) | 32 (value target = game outcome) | no | not tuned this pass, see below |
| `bufferCapacity` | 2000 games | no | starting-point default |
| `evalIntervalIterations` / `evalGamesPerSide` / `evalSimulations` | 10 / 20 / 150 | no | starting-point defaults |

A fresh `./build/train 800 checkpoint.bin` with the code as committed has,
on the Grid 2 evidence, roughly 1-in-3 odds of reproducing seed 303's
convergence and 2-in-3 odds of a partial or complete failure. Pass a
third argument (a seed) to reproduce a specific outcome from this
document exactly, e.g. `./build/train 800 checkpoint.bin 303`.

### `tdSteps` (bootstrapping) — not tried, and why it is still the top candidate

Not tried this pass either (five runs of budget in the first attempt, six
plus six in the two grids requested by the coordinator — every run slot
was spoken for by the seed-variance and search-budget questions, which
were the more urgent open questions). This is a real gap, not a judgment
that it wouldn't matter. If anything, the value-head/policy-head probe
finding sharpens the case for trying it: the value head is *already*
correct (right signs, sensible magnitudes) under the current `tdSteps=32`
regime, where the value target is just the game outcome — so lowering
`tdSteps` to turn on real bootstrapping would not be fixing a broken
value head, it would be asking whether a denser, earlier value signal
changes what the *policy* head learns to prioritize during training
(since the policy loss and value loss share the trunk of the network).
Whether that helps is a real open question, not a predictable win.

## What to try next, and what to spend the next hour on

In order of how directly each follows from what was actually measured
this pass:

1. **The `temperatureMoves=2` vs. `6` ablation at seed 303, `numSimulations=100`, 800 iterations.**
   This is the single most informative next run and was identified above
   as the missing cell. It directly answers whether `temperatureMoves=6`
   is doing real work or whether seed 303 was simply a lucky seed
   regardless of it — a question this document currently cannot answer
   and should not paper over.
2. **More `gamesPerIteration`.** The diagnosed failure is a single rare
   tactical pattern (a two-in-a-row that must be blocked, arising a
   specific number of plies into a specific opening) that the policy head
   does not reliably learn. If self-play only occasionally produces the
   position, the training buffer only occasionally contains a corrective
   example for it, and a config-and-seed-dependent coin flip on whether
   that example appears (and appears often enough to be sampled and
   trained on before the buffer rolls it out) is a plausible mechanism for
   the seed variance observed throughout every grid in this document. More
   games per iteration increases the odds that the position is generated
   and retained, independent of `temperatureMoves`.
3. **`tdSteps` lowered (e.g. to 5).** Discussed above; genuinely open,
   and specifically interacts with the shared trunk between the value and
   policy heads in a way the current evidence cannot predict.
4. **More `trainStepsPerIteration` or a larger hidden width.** The
   policy head's failure could be a capacity or fitting problem rather
   than (or in addition to) a data problem — worth ruling in or out once
   the data-side candidates (2) are tried, since more gradient steps on
   data that never contains the corrective example won't help, but more
   steps on data that does contain it (rarely) might make better use of
   the rare signal.

If forced to spend the next hour on exactly one: **(2), more
`gamesPerIteration`.** It is the cheapest to test (no algorithm or
architecture change, one number in `apps/train.cpp`), it is directly
motivated by the localized failure (a rare position under-represented in
training data) rather than by a mechanism this pass already tested and
found wanting (search budget), and unlike (1) it does not merely explain
the existing evidence — it is a real, independent lever that either
raises every seed's hit rate or it does not.

## Global constraints observed

No changes were made to `src/mcts.cpp`, `src/network.cpp`,
`src/targets.cpp`, or `src/selfplay.cpp`. `include/mz/replay_buffer.hpp`
and `src/replay_buffer.cpp` were changed only to add an optional,
default-preserving seed parameter for reproducibility — no sampling
*behavior* changed for any existing caller. `include/mz/selfplay.hpp`'s
`SelfPlayConfig` defaults were changed (`numSimulations`,
`temperatureMoves`), as explicitly permitted, with the rationale recorded
both here and as comments at the point of definition. Legality was never
masked below the search root at any point, in any of the 23 runs — search
below the root was free to (and did, extensively — see the depth-18
finding) wander into branches the real game does not allow. The
evaluation harness (`evaluateAgainstMinimax`, `evalGamesPerSide=50`) was
never weakened.
