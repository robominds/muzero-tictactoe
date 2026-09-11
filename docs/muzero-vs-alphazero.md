# MuZero vs. AlphaZero: the comparison

This project (`mz/`, this repository) and its sibling
[`alphazero-tictactoe`](../../alphazero-tictactoe) (`az/`) implement the same
game with the same board representation, the same evaluation harness, and
deliberately similar code shape, so that every place the two diverge is a
direct consequence of one fact: **AlphaZero's search is allowed to consult
the real rules of the game, and MuZero's is not.** AlphaZero calls
`az::Board::applyMove` inside its tree; MuZero calls
`MuZeroNetwork::recurrentInference` — a learned function with no board
underneath it at all.

For a guided, line-by-line walkthrough of this codebase with short inline
"vs AlphaZero" notes at each point of divergence, see
[`docs/algorithm-explained.md`](algorithm-explained.md). This document does
not repeat that walkthrough; it is the standalone, side-by-side treatment of
*why* each divergence exists and *what it costs*, with both codebases' real
source cited throughout.

## Summary table

| Dimension | AlphaZero | MuZero |
|---|---|---|
| Game dynamics inside search | The real board, stepped by `Board::applyMove` (`az/src/mcts.cpp:83`) | A learned model, stepped by `MuZeroNetwork::recurrentInference` (`mz/src/mcts.cpp:155-156`) |
| State representation | The actual board (`Node::board`, `az/include/az/mcts.hpp`) | An opaque latent vector with no decoder (`mz/include/mz/network.hpp:17-21`) |
| Legality knowledge in the tree | Every node: `node.board.legalMoves()` gates expansion and selection at every depth (`az/src/mcts.cpp:35,52,60`) | Root only: `MCTS::maskAndRenormalize` (`mz/src/mcts.cpp:62-76`); every other node expands all nine actions (`mz/src/mcts.cpp:164-167`) |
| Terminal detection | Search stops at a terminal board and returns the true result (`az/src/mcts.cpp:73-76`) | None; the tree has no concept of "game over" and searches straight through it |
| Reward | Not modeled; only a final win/draw/loss value | A learned reward head on every edge (`mz/include/mz/network.hpp:41`), almost always targeting zero on this domain |
| Value target | Final game outcome, `z ∈ {-1, 0, +1}` (`az/src/selfplay.cpp:29-38`) | n-step bootstrap from search values and rewards (`mz/src/targets.cpp:17-33`) |
| Search Q normalization | None needed — Q is already a bounded `tanh` value | `MinMaxStats` rescales Q into [0, 1] every simulation (`mz/include/mz/mcts.hpp:19-34`) |
| PUCT exploration term | Fixed constant `cPuct` (default 1.5) (`az/include/az/mcts.hpp:25`, `az/src/mcts.cpp:62`) | Visit-count-dependent formula with `c1=1.25`, `c2=19652` (`mz/include/mz/mcts.hpp:14-16,60`) |
| Replay storage | Flat per-position `TrainingExample` (`az/include/az/network.hpp:13-17`) | Whole trajectories, `GameHistory`, in a ring buffer (`mz/include/mz/game_history.hpp`, `mz/include/mz/replay_buffer.hpp`) |
| Network count | One, shared trunk plus two heads (`az::Network`, `az/include/az/network.hpp:19-45`) | Three: representation `h`, dynamics `g`, prediction `f` (`mz::MuZeroNetwork`, `mz/include/mz/network.hpp:108-110`) |
| Can stale training data be improved? | No — targets are facts about a finished game | Yes, in principle (Reanalyze) — targets are the network's own opinion, and are stored in a form (`ReplayBuffer::replaceSearchTargets`) built to support refreshing them; not implemented here |

## Given vs. learned dynamics

AlphaZero's tree walks the real board. `MCTS::simulate` in
`az/src/mcts.cpp:73-90` calls `node.board.applyMove(move)` (line 83) to
produce each child's board, and `Board::applyMove` (`az/src/board.cpp:47`)
is exact, deterministic tic-tac-toe: no approximation, no error to
accumulate.

MuZero cannot do this, because it is never given the rules. Every non-root
step in `MCTS::run` (`mz/src/mcts.cpp:119-224`) calls
`network_.recurrentInference(parent->latent, lastAction)` at line 155-156 —
a learned function of a learned representation, with no reference to a real
board anywhere in the call. `MuZeroNetwork::recurrentInference` is declared
in `mz/include/mz/network.hpp:63` and documented there as reaching every
non-root node "with no reference to the real board at all."

The cost is direct: everything AlphaZero's search gets for free — exact
transitions, exact terminal states, exact legality — MuZero's search has to
learn from self-play data, approximately, and the approximation error
compounds with search depth (see "Root-only legality" below, and
`docs/results.md`'s `latent_probe` findings).

## The latent state, and the loss of interpretability

AlphaZero's tree nodes hold an actual `Board` (`az/include/az/mcts.hpp`,
`Node::board`). You can print one, hand it to `minimax`, or show it to a
person, at any depth in the tree.

MuZero's tree nodes hold a `std::vector<float>` of length 32
(`MuZeroNetwork::kLatentSize`, `mz/include/mz/network.hpp:30`) with no
decoder back to a board anywhere in the system. The header is explicit
about this being deliberate, not an oversight (`mz/include/mz/network.hpp:17-21`):

> There is deliberately NO decoder from latent back to a board. Nothing
> here can look at a latent and say which position it is. A latent only
> has to support good policy, value and reward predictions — it is never
> asked to reconstruct the observation.

That constraint is exactly why this project needed a tool AlphaZero has no
use for: `latent_probe`. Since nobody — human or code — can look at a
MuZero latent and name the position it represents, the only way to check
whether the dynamics model has actually learned the game is indirect: roll
`g` forward `k` steps from a real position, and separately take the real
board `k` actions later through a fresh `h`+`f` pass, then compare the two
predictions. That's `latent_probe`'s entire job (see `docs/results.md`'s
findings from it — mean policy distance grows from 0.24 at k=1 to 0.55 at
k=6, even on the one converged checkpoint). AlphaZero needs no equivalent
tool, because at any depth it can just print `node.board` and look at it.

## Root-only legality

AlphaZero's board knows which moves are legal at every node it visits, and
the search always asks: `Board::legalMoves()` is called at expansion
(`az/src/mcts.cpp:35`, inside `MCTS::expand`), at selection
(`az/src/mcts.cpp:52,60`, inside `MCTS::selectChild`), and at the top level
of `MCTS::run` (`az/src/mcts.cpp:97`).

MuZero's search asks the real board exactly once, at the root: `maskAndRenormalize`
(`mz/src/mcts.cpp:62-76`) masks `initial.policy` down to `legalActions` and
renormalizes, and that masked, (optionally) noised prior is the only place
legality enters the whole search (`mz/src/mcts.cpp:126-136`, comment at
line 127: "The one and only place legality enters the search"). Every node
below the root expands all nine actions unconditionally
(`mz/src/mcts.cpp:164-167`):

```cpp
// Below the root, every one of the nine actions gets a child.
// There is no legality oracle here and no terminal detection...
for (int a = 0; a < 9; ++a) {
    node->children[a] = std::make_unique<Node>();
    node->children[a]->prior = step.policy[a];
}
```

`mz/tests/test_mcts.cpp:156-182`,
`test_tree_grows_below_a_root_with_only_one_legal_move`, makes the
consequence concrete: it fills the board to a single legal move at the
root, then asserts the search still reaches `maxDepth >= 2` — i.e., the
tree grows *past* a position that, played for real, has no legal moves at
all, because nothing below the root knows that.

**This wastes search on this domain, measurably.** `docs/results.md`
reports search reaching depth 18–19 during `diag_eval` at 400 simulations,
on a board that cannot legally be more than 9 plies deep — "the median
simulation is illegal by the time it terminates." And the headline
convergence comparison in `docs/results.md` is exactly this cost made
visible: AlphaZero reached `draws=40 losses=0` in 20 iterations
(`az/README.md`); this project's best configuration converged on 1 of 3
seeds after 800 iterations, and 0 of 3 seeds at every other configuration
tried up to 1500 iterations. See "The headline result" below for the full,
honest statement of that gap.

## Terminal detection

`MCTS::simulate` in AlphaZero checks `node.board.isTerminal()` first thing
(`az/src/mcts.cpp:74`) and returns the true result — `0.0f` for a draw,
`-1.0f` otherwise, from the perspective of the player about to move into
that terminal state — without ever expanding past it. Search simply cannot
continue past the end of a real game.

MuZero's tree has no terminal concept anywhere. There is no board inside
the tree to call `isTerminal()` on, and no analogous check anywhere in
`mz/src/mcts.cpp`. The search keeps expanding nodes past what would be the
end of the game (see "Root-only legality" above for the depth-18 evidence),
relying entirely on the reward head predicting a strongly negative or
positive terminal payoff and the value head predicting an extreme value, so
that continuing to play into (and past) a lost position scores badly enough
in the PUCT comparison to be avoided. Nothing structurally prevents the
search from wandering there anyway if those heads are wrong, and
`docs/results.md`'s depth findings show it routinely does.

## The reward head

AlphaZero has no reward head and needs none: `az::Prediction`
(`az/include/az/network.hpp:8-11`) is just `{policy, value}`. Its search
gets the true final outcome directly from `Board::outcome()` at a terminal
node, so there is nothing for a reward function to predict.

MuZero's `recurrentInference` returns a `reward` alongside the next latent,
policy, and value (`mz/include/mz/network.hpp:39-44`), and `MCTS::run`
records it on every edge (`node->stats.reward = step.reward;`,
`mz/src/mcts.cpp:159`) and folds it into backup:
`edgeQ = childReward + discount * -childValue` (`mz/src/mcts.cpp:28-30`,
`mz/include/mz/mcts.hpp:53-57`). On tic-tac-toe the reward is zero at every
transition except the one that ends the game (documented at
`mz/include/mz/game_history.hpp:22`, and the design spec's "Reward on this
domain" section), so in practice the head spends nearly all of training
predicting zero and is almost never exercised.

It is still structural, not optional, for two reasons. First, MuZero's
general formulation assumes rewards at every step (Atari has one per
frame); a board game like this is a corner case of the same algorithm, not
a different one, and dropping the head would be dropping part of the
algorithm rather than specializing it. Second, even on this domain the head
carries real signal at exactly the transitions that matter: a network that
cannot predict "this move ends the game, and in whose favor" cannot search
correctly over its own model at the one place accuracy is most needed. And
that is precisely where `docs/results.md`'s `latent_probe` measurement
finds the head weakest: "Terminal reward prediction: mean |error| = 0.7682
over 1153 transitions" — on rewards restricted to `{-1, 0, +1}`, an error
of 0.77 is large, on the one checkpoint in the whole project that plays
perfectly against minimax. The reward head is real machinery doing an
important job it does only partially well, not vestigial.

## Bootstrapped vs. outcome value targets

AlphaZero's value target is the actual game outcome. `playSelfPlayGame`
(`az/src/selfplay.cpp:16-40`) plays a whole game to termination, reads
`Board::outcome()`, and assigns every stored position a `z` of `+1`, `0`,
or `-1` depending on whether the player to move there ultimately won, drew,
or lost (lines 29-38). That target is a fact about a finished game. It
cannot become more correct later — a better network training on the same
stored example gets the identical target, because nothing about who
actually won a game already played can change.

MuZero's target is `bootstrappedValue` in `mz/src/targets.cpp:17-33`:

```
z_t = sum_{i in [t, t+n)} (-1)^(i-t) * discount^(i-t) * reward_i
      + (-1)^n * discount^n * searchValue_{t+n}
```

— an n-step sum of observed rewards plus a bootstrap off `searchValues[t+n]`,
which is `mz::GameHistory`'s stored MCTS root value from when the game was
played (`mz/include/mz/game_history.hpp:19,27`). That value came from the
network's own search at self-play time. It is the network's *opinion*, not
a fact about the outcome, so a stronger network produces a better
opinion — which is exactly the property that makes Reanalyze possible (see
below). With `tdSteps=32` (past any possible game length on a 9-ply board),
the bootstrap term always falls out of range and the sum collapses to the
plain game outcome, `docs/results.md`'s configuration used throughout this
project's tuning pass — so in practice, as configured here, the two
projects' value targets currently coincide in effect even though they
differ in mechanism. Lowering `tdSteps` is the one dimension explicitly
flagged in `docs/results.md` as untried and still open.

## Trajectory vs. position storage

AlphaZero flattens self-play straight into independent positions.
`playSelfPlayGame` (`az/src/selfplay.cpp:16-40`) accumulates a
`PendingExample` per ply and, once the game ends, converts each into a
`TrainingExample` (`az/include/az/network.hpp:13-17`) — an encoded board, a
target policy, and a scalar target value — and discards the trajectory
structure entirely. `ReplayBuffer::sampleBatch`
(`az/include/az/replay_buffer.hpp:15`) then draws these independent,
already-labeled positions uniformly.

MuZero cannot do this, because both of its training targets read *forward*
from a position rather than being computed once and stored flat. The
K-step unroll needs the actions actually played after a position; the
n-step value target needs `rewards` and `searchValues` at later plies in
the same game. `mz::GameHistory` (`mz/include/mz/game_history.hpp:23-31`)
therefore stores a whole trajectory — `observations`, `actions`,
`searchPolicies`, `searchValues`, `rewards`, all indexed by ply — and
`ReplayBuffer` (`mz/include/mz/replay_buffer.hpp:13-53`) is a ring buffer
of these whole `GameHistory` objects, not of positions. `samplePositions`
(line 36) draws a `(gameIndex, position)` pair, and `makeUnrolledSample`
(`mz/src/targets.cpp:48`) is what reads forward from that position to build
a training sample — something a flattened `TrainingExample` has no way to
support, because the later actions, rewards, and search values it would
need to read are simply gone by the time it exists.

## The two PUCT formulas

AlphaZero (`az/src/mcts.cpp:62-63`):

```cpp
float u = cPuct_ * node.priors[m] * std::sqrt(static_cast<float>(totalVisits) + 1e-8f)
          / (1 + node.visitCounts[m]);
```

with `cPuct` a fixed constant, defaulted to `1.5f`
(`az/include/az/mcts.hpp:25`, and again in `az/src/mcts.cpp:8-11`).

MuZero (`mz/src/mcts.cpp:32-35`, `explorationTerm`):

```cpp
float pbC = std::log((parentVisits + kPbCBase + 1.0f) / kPbCBase) + kPbCInit;
return prior * std::sqrt(parentVisits) / (1.0f + childVisits) * pbC;
```

with `kPbCInit = 1.25f` and `kPbCBase = 19652.0f`
(`mz/include/mz/mcts.hpp:15-16`), the paper's appendix-B constants in place
of a single tuned `cPuct`. Structurally the two formulas are the same
prior-weighted exploration bonus scaled by `sqrt(parentVisits)/(1+childVisits)`;
MuZero's version additionally grows that scale slowly as `parentVisits`
increases, through the `log` term.

Honestly: at tic-tac-toe's simulation counts (dozens to a few hundred per
move), that log term barely moves. The header comment on `kPbCBase` says so
directly (`mz/include/mz/mcts.hpp:12-14`): "`c2` is large enough that the
log term barely moves at tic-tac-toe's simulation counts; it is here
because the mechanism, not its magnitude, is the point." At
`parentVisits=100`, `log((100+19653)/19652) ≈ 0.0050`, added to `c1=1.25` —
a fraction of a percent effect. The formula is implemented faithfully
because the point of this project is to implement the real mechanism, not
because it changes tic-tac-toe outcomes; AlphaZero's fixed `cPuct=1.5` and
MuZero's `c1=1.25` land in the same ballpark for the same reason — both are
just the constant exploration weight that matters at this scale.

One further, related asymmetry: MuZero needs `MinMaxStats`
(`mz/include/mz/mcts.hpp:18-34`) to normalize Q into `[0,1]` before this
comparison (`mz/src/mcts.cpp:107`, `stats.normalize(edgeQ(...))`), because
its Q mixes a learned reward of unknown scale with a value; AlphaZero's Q
(`az/src/mcts.cpp:61`) is already a bounded `tanh` output and is compared
to the prior directly, no normalization step at all.

## One network vs. three

AlphaZero is one network with a shared trunk and two heads: `az::Network`
(`az/include/az/network.hpp:19-45`) has a single hidden layer (`w1_`,
`b1_`, 18 → 64) feeding both a policy head (`wPolicy_`, 64 → 9) and a value
head (`wValue_`, 64 → 1). One `predict` call is all any position ever needs; `Network::trainStep`
(`az/src/network.cpp:54`) runs a single forward and backward pass per
example (loop starts at line 64), with no recurrence at all.

MuZero is three: representation `h` (`hFc1_`, `hFc2_`), dynamics `g`
(`gFc1_`, `gFc2_`, `gReward_`), and prediction `f` (`fFc1_`, `fPolicy_`,
`fValue_`) — eight `Dense` layers total, enumerated in
`MuZeroNetwork::LayerId` (`mz/include/mz/network.hpp:71-75`).

The training consequence is direct. AlphaZero's `trainStep`
(`az/src/network.cpp:54`) runs one forward and one backward pass per
example. MuZero's `trainStep` (`mz/src/network.cpp:160` onward) runs `h` once,
then `g` and `f` repeatedly across a K-step unroll in a single forward
pass, and backpropagates through all of it in one reverse loop
(`mz/src/network.cpp:229` onward) — backprop-through-time, because
`g` is applied to its own previous output K times inside one loss. That
recurrence is exactly why `mz::Dense::backward` had to *accumulate* into
its gradient buffers rather than overwrite them
(`mz/src/mlp.cpp:48-51`, comment: "accumulate, never assign"): each of the
K applications of the same `g` layer contributes its own gradient to the
same shared weights within a single `trainStep` call, and a plain
assignment would silently discard every application but the last. Two
further scaling rules exist only because of this recurrence and have no
AlphaZero equivalent: every unroll step past 0 is scaled by `1/K`
(`mz/src/network.cpp:178-179`), and the gradient flowing back into
dynamics from its latent input is halved at every step
(`kHalfGradient = 0.5f`, `mz/include/mz/network.hpp:87`,
`mz/src/network.cpp:285`) to keep gradient magnitude from compounding
across the recurrence.

## Reanalyze: a capability AlphaZero structurally cannot have

Reanalyze re-runs search on already-stored positions using the *current*
network, and replaces the stale targets recorded when the trajectory was
originally played. It is not implemented in this project, but three
architectural decisions leave room for it (matching the design spec's
"Reanalyze Readiness" section) — and this repository verifiably has all
three:

1. **The replay buffer stores whole trajectories with raw observations**,
   not flattened positions. `mz::GameHistory` (`mz/include/mz/game_history.hpp:23-31`)
   keeps `observations[t]` for every step, which is exactly the input a
   fresh search would need to re-run.
2. **`searchPolicies` and `searchValues` are replaceable in place.**
   `ReplayBuffer::replaceSearchTargets` (`mz/include/mz/replay_buffer.hpp:38-46`)
   is documented as a "Reanalyze hook": it overwrites a stored trajectory's
   search targets while leaving `observations`, `actions`, and `rewards`
   untouched, because those are "facts about what happened" that "never go
   stale," whereas "search policies and values are the network's opinions
   at the time, and a stronger network can improve them" (comment at
   `mz/include/mz/replay_buffer.hpp:39-43`).
3. **`makeUnrolledSample` is a pure function**, re-invocable over refreshed
   values. `mz/include/mz/targets.hpp:41-46` states this directly: it is "a
   function of the trajectory and the config only. That is deliberate —
   Reanalyze works by refreshing a trajectory's stored search targets and
   calling this again, which only works if nothing else here carries
   state." `mz/src/targets.cpp:48` implements it with no side effects and
   no state beyond its arguments.

Bootstrapped targets are the precondition that makes any of this possible
at all. Reanalyze only has something to gain by re-running search: since
`bootstrappedValue` (`mz/src/targets.cpp:17-33`) is built from
`searchValues`, and a `searchValue` is a snapshot of what the network
believed about a position at the moment it was played, re-running MCTS with
a *later*, presumably-better network on that same stored observation
produces a different, presumably more accurate `searchValue` — genuinely
new information the buffer did not have before. Reanalyze is the mechanism
that goes and collects that improvement instead of leaving it stranded in
old data.

AlphaZero cannot have this, structurally, not as a missing feature but as a
direct consequence of what its value target *is*. `playSelfPlayGame`
(`az/src/selfplay.cpp:29-38`) assigns `z` from `Board::outcome()` after the
game has already ended. That is a fact about the past: whichever player
actually won that specific played-out game did or did not win, permanently,
and no future network — however much stronger — can change what already
happened. Re-running search on an old `TrainingExample` and getting a new
value would just be a different, unrelated opinion about a hypothetical
game that never happened; it would not be a way to *correct* the stored
`z`, because there is nothing wrong with `z` to correct. This is exactly
the asymmetry the design spec calls out directly: "AlphaZero cannot do
this. Its value targets are final game outcomes, which are facts about the
past and cannot be improved by a better network. MuZero's targets are
bootstrapped from its own search, so a stronger network makes old data more
valuable" (`docs/superpowers/specs/2026-09-10-muzero-tictactoe-design.md:308-312`).

## The headline result: the measured cost of root-only legality

`docs/results.md` is the full account; this is the summary specific to the
MuZero/AlphaZero difference.

**The spec's first success criterion — converging to drawing play against
minimax — was not met by this MuZero implementation.** Across 23 training
runs, exactly one converged: `numSimulations=100`, `temperatureMoves=6`,
800 iterations, seed 303, `wins=0 draws=100 losses=0`. The other two seeds
at that identical configuration gave `draws=50 losses=50` and
`draws=0 losses=100`. No configuration tried converged on all three seeds
of any cell tested. The sibling AlphaZero project reached
`draws=40 losses=0` in 20 iterations (`az/README.md`).

That gap is real, but an "iteration" is not the same unit of work in the
two projects, and the comparison should not be read as more precise than
it is. Nor is it measured to the same standard: `az/README.md` reports a
single run, with no seed control and no grid, where this document's MuZero
numbers come from a matched 3-seed-per-cell sweep. The AlphaZero side of
this comparison is one data point, not an average or a best-of. Reading
both projects' `apps/train.cpp` directly (as
`docs/results.md`'s "Comparison with the AlphaZero sibling project"
section does):

| | AlphaZero | MuZero (this project, best found) |
|---|---|---|
| Self-play games/iteration | 25 | 25 |
| MCTS simulations/move (self-play) | 50, over the real rules | 100, over a learned model |
| Gradient steps/iteration | 20 | 40 |
| Batch size | 32 | 64 |

MuZero does roughly 4x the gradient-step throughput of AlphaZero per
iteration, and 800 MuZero iterations amount to something like 80x
AlphaZero's total gradient-step-samples — yet converged on only 1 of 3
seeds, where AlphaZero converged in 20 iterations of its smaller budget
(on the single run reported in `az/README.md` — see the caveat above).
The imprecision is real: the two self-play loops, network
architectures (MuZero has a representation network and a reward head with
no AlphaZero equivalent), and target constructions (K-step unrolling)
differ enough that iteration count is not a unit-comparable currency
between the two projects. What *is* precise is the mechanism: AlphaZero's
search never leaves the real board, so its self-play tail reliably
rediscovers any corrective line; MuZero's search increasingly imagines
positions the real game cannot reach, and its self-play tail can lock onto
a wrong belief indefinitely because it never revisits the position that
would correct it.

Two further findings from `docs/results.md` are specifically about this
divergence and belong here:

- **More search budget made MuZero's results worse, not better.** A
  matched-seed comparison (same three seeds, `temperatureMoves=6`, 800
  iterations, only `numSimulations` varied) went from 1/3 converged at 100
  simulations to 0/3 converged at 300. Deeper trees accumulate more
  learned-model error rather than resolving more of the real game, because
  below the root the model has no legality oracle to keep it honest.
  AlphaZero has no equivalent failure mode: its deeper nodes are the exact
  board, so more search simulations can only ever add real information,
  never manufacture false confidence out of accumulated model drift.
- **Search reaches depth 18–19 on a nine-square board.** Below the root,
  MuZero's branching factor never shrinks — it is always nine, because
  nothing marks a square as already played. AlphaZero's branching factor
  shrinks with every real ply — 9, 8, 7, 6, ... — because `legalMoves()` is
  consulted at every node (`az/src/mcts.cpp:35,52,60`). The same simulation
  budget therefore explores a genuinely smaller, always-legal space in
  AlphaZero and an unbounded, mostly-illegal space in MuZero.

The one converged run is real evidence the implementation is correct end to
end: at that checkpoint, the value head reports correct signs and sensible
magnitudes in both directions, the same checkpoint's MCTS visit
distribution puts 74-80% of its mass on the correct blocking move at the
decisive ply (`docs/results.md`'s `diag_eval` transcripts), and it draws
100/100 games against a perfect minimax opponent as both X and O. The
failure across the other 22 runs is a reliability problem — the policy
head not learning a rare tactical pattern consistently enough, worsened by
search that cannot fall back on the real rules below the root to correct
it — not a structural defect in the algorithm as implemented.

## Deviations from the paper

These are simplifications made deliberately for this domain, not omissions
from the algorithm — and the point of listing them here is so a reader of
this codebase never mistakes one of them for what the real algorithm does.

- **Scalar value and reward heads, `tanh`-bounded, instead of the
  categorical distribution over a 601-bin support with its invertible
  scaling transform.** The paper predicts value and reward as a softmax
  over 601 bins spanning a wide numeric range, using the transform
  `h(x) = sign(x)(sqrt(|x|+1) - 1) + eps*x` to compress large returns (this
  matters enormously for Atari, where returns can be in the thousands and
  a scalar regression target is numerically unstable and hard to learn).
  Tic-tac-toe's returns are exactly `{-1, 0, +1}` in every case; a `tanh`
  scalar head already covers that entire range exactly, so the 601-bin
  machinery would add substantial code and no benefit on this domain. Both
  `MuZeroNetwork`'s value output and its reward output (`gReward_`,
  `fValue_` in `mz/include/mz/network.hpp:108-110`) are single `tanh`
  scalars.
- **SGD instead of Adam with the paper's learning-rate schedule.** The
  paper trains with Adam and a decaying learning-rate schedule tuned for
  hundreds of millions of self-play frames. This project uses plain SGD
  with a fixed learning rate (`applySgd`, `mz::Dense`), matching the
  sibling AlphaZero project's approach, which was chosen there for the
  same reason: at this problem's scale, momentum and adaptive per-parameter
  learning rates are not the bottleneck, and keeping both projects' optimizers
  identical keeps the comparison about the algorithm, not the optimizer.
- **No Reanalyze.** The real algorithm periodically re-runs search on
  stored trajectories with the current network to refresh stale targets,
  meaningfully improving sample efficiency at scale. It was excluded here
  as an explicit non-goal (see the section above) — it adds real
  complexity (a second self-play-like loop, scheduling between fresh and
  reanalyzed data) for a benefit that mostly shows up in large,
  long-running training regimes, not in a project sized to a board with 9
  squares and training runs measured in seconds to minutes. The
  architecture is left ready for it (see above) rather than closed off.
- **Small MLPs — 64-unit hidden layers and a 32-unit latent — instead of
  the paper's deep residual towers.** The paper's networks are large
  convolutional/residual stacks sized for board games like Go and Chess
  and for raw Atari frames. `MuZeroNetwork`'s `h`, `g`, and `f` are each
  one or two `Dense` layers of width 64 (design spec: `mz/include/mz/network.hpp:30-31`,
  `kLatentSize=32`, `kHiddenSize=64`). Tic-tac-toe's 18-float observation
  and 9-action space do not need — and could not usefully exploit — a
  residual tower; a network that size would be many times larger than the
  entire state space it is trying to represent.
