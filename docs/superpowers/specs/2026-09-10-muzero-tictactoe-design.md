# MuZero-Style Tic-Tac-Toe Player — Design

## Purpose

A from-scratch, dependency-free C++ implementation of the MuZero
algorithm — self-play and PUCT search conducted entirely inside a
*learned* model of the game — applied to tic-tac-toe, built as a
learning project and as a direct point of comparison against the
sibling project `alphazero-tictactoe`.

The goal is to understand every moving part by implementing it
directly, and to make the AlphaZero-to-MuZero delta concrete: what
changes when the search is no longer allowed to consult the rules of
the game.

## Success Criteria

- Training converges to optimal play: the trained agent draws (never
  loses) against a perfect minimax player, as X and as O.
- The learned dynamics function demonstrably tracks the real game:
  `latent_probe` shows that rolling dynamics forward k steps from a
  position yields a latent whose predicted policy and value stay close
  to a fresh representation pass on the real board those actions reach.
- Every component (mlp, network, mcts, game history, replay buffer,
  targets, self-play, evaluation, CLI) is testable independently.
- No external dependencies beyond the C++ standard library.
- The documentation set explains the algorithm and its differences from
  AlphaZero well enough to learn from without reading the papers first.

A note on convergence: MuZero is expected to need materially more
iterations than AlphaZero on this problem. That gap is a *result to
measure and document*, not a defect — see "Root-Only Legality" below.

## Non-Goals

- Generalizing beyond tic-tac-toe.
- MuZero Reanalyze. Not implemented in this spec, but the architecture
  is deliberately shaped so it can be added without restructuring:
  the replay buffer stores whole trajectories rather than flattened
  positions, and target construction is a separate, re-runnable step.
  See "Reanalyze Readiness".
- The categorical value/reward representation from the paper (a
  softmax over a 601-bin support with an invertible scaling
  transform). Scalar `tanh` heads are used instead. This is a
  deliberate, documented deviation — see "Deviations From the Paper".
- The AlphaZero arena/network-promotion step, matching the sibling
  project.
- GPU acceleration, batched or parallel MCTS, performance tuning.

## Architecture

A shared static library (`libmz`) of independent, single-purpose
components, consumed by three executables and two diagnostic tools.

```
train ------\
evaluate ----+--> libmz
play_cli ---/          board, minimax          (rules: root + real-game stepping only)
                       mlp                     (Dense layer: forward/backward/SGD)
diag_eval ---\         network                 (representation, dynamics, prediction)
latent_probe -+->      mcts                    (PUCT over latent states)
                       game_history            (one trajectory)
                       replay_buffer           (ring buffer of trajectories)
                       targets                 (trajectory + index -> unrolled sample)
                       selfplay, eval
```

### Components

**`board`, `minimax`**
Carried over from `alphazero-tictactoe` essentially unchanged: 3x3
grid, legal moves, win/draw/ongoing detection, player-relative
18-float encoding, and an exhaustive perfect-play solver used only for
evaluation.

Keeping them identical is the point of the comparison. The rules are
sitting right there; MuZero is simply not permitted to consult them
inside the search tree.

**`mlp`**
The hand-written differentiation primitive. A `Dense` layer owns
weights, biases, and their accumulated gradients, and provides:
- `forward(input)` — computes the output and caches whatever the
  backward pass needs.
- `backward(dOutput)` — returns the gradient with respect to the
  input and *accumulates* into the weight/bias gradients.
- `zeroGrad()` / `applySgd(learningRate)`.

Accumulation rather than assignment is what makes
backprop-through-time work: the dynamics network is applied K times in
a single loss, and each application must contribute to one shared
gradient.

Uses `std::vector` for weights rather than the fixed-size
`std::array` shapes in the AlphaZero network, because three networks
of different dimensions share this primitive.

**`network`** — `MuZeroNetwork`
The three functions of MuZero, each a small MLP built from `Dense`:

| function | maps | sizes |
|---|---|---|
| representation `h` | observation -> latent | 18 -> 64 (ReLU) -> 32 |
| dynamics `g` | (latent, action) -> (latent, reward) | 41 -> 64 (ReLU) -> 32 and -> 1 (tanh) |
| prediction `f` | latent -> (policy, value) | 32 -> 64 (ReLU) -> 9 logits and -> 1 (tanh) |

The action reaches dynamics as a 9-element one-hot vector
concatenated onto the latent. Every latent produced by `h` or `g` is
min-max normalized to [0, 1] across its 32 units, per the paper's
appendix; without it, repeated application of `g` lets latent
magnitudes drift.

Public interface, which is all MCTS ever sees:
- `initialInference(observation) -> {latent, policy, value}` (`h` then `f`)
- `recurrentInference(latent, action) -> {nextLatent, reward, policy, value}` (`g` then `f`)
- `trainStep(batch of UnrolledSample, learningRate) -> Losses`
- `save(path)` / `load(path)`

Note what is absent: there is no decoder from latent back to a board.
Nothing in the system can look at a latent and say which position it
is. That is not an omission — it is MuZero's defining constraint. The
latent is only ever required to support good policy, value, and reward
predictions, never to reconstruct the observation.

**`mcts`**
PUCT search over latent states.
- The root is expanded via `initialInference` on the real board's
  encoding. Only here are legal moves known: illegal actions are
  masked out of the root prior and renormalized, and Dirichlet noise
  is mixed into what remains.
- Every non-root node expands **all nine actions** via
  `recurrentInference`. The tree has no legality oracle and no
  terminal detection.
- Each edge carries the reward its dynamics step predicted. Backup
  propagates `G = reward + discount * (-childValue)`, negating at each
  ply because the players alternate. Discount is 1.0 for a board game.
- `MinMaxStats` tracks the running min and max of Q values seen in the
  tree, and Q is normalized into [0, 1] before entering the PUCT
  comparison. AlphaZero can skip this because a `tanh` value head is
  already bounded and directly comparable to a prior; MuZero's Q
  mixes in predicted rewards of unknown scale, so the exploration term
  needs a normalized quantity to trade against.
- Exploration uses the paper's formula rather than a fixed constant:

  `score = normalizedQ + P * (sqrt(N_parent) / (1 + N_child)) * (c1 + log((N_parent + c2 + 1) / c2))`

  with `c1 = 1.25`, `c2 = 19652`. The log term makes exploration grow
  slowly with visit count instead of staying fixed.
- Returns a visit distribution over the nine actions and a selected
  move. Move selection at the root restricts to legal moves.

**`game_history`**
One complete self-play trajectory:

```
observations[t]     encoded board at step t
actions[t]          action taken at step t
searchPolicies[t]   MCTS root visit distribution at step t
searchValues[t]     MCTS root value at step t   (bootstrap source)
rewards[t]          reward received after action t, from the
                    perspective of the player to move at step t
```

This replaces AlphaZero's flat `TrainingExample`. A K-step unroll and
an n-step bootstrap both need to read *forward* from a position, which
a flattened position cannot support.

**`replay_buffer`**
Fixed-capacity ring buffer of `GameHistory` objects. Sampling draws a
(trajectory, position) pair uniformly. Exposes accessors that allow a
future pass to overwrite a trajectory's stored `searchPolicies` and
`searchValues` in place. No prioritization.

**`targets`**
Pure function from `(GameHistory, position, unrollSteps, tdSteps,
discount)` to an `UnrolledSample`: the observation at `position`, the
K actions actually played from it, and aligned targets: K+1 value
targets and K+1 policy targets (steps 0..K), plus K reward targets
(steps 1..K). Step 0 has no incoming transition, so it has no reward
target.

Value targets use the n-step bootstrap:

```
z_t = sum over i in [0, n) of  discount^i * r_{t+i}
      + discount^n * v_{t+n}
```

with a sign flip at every ply, since `v` and `r` are each stored from
the perspective of whoever was to move. `tdSteps` is configurable;
setting it beyond the maximum game length collapses the expression to
the plain game outcome, which is what board-game MuZero uses. The
general form is implemented so the mechanism is visible; the default
reproduces the familiar behavior.

Unroll steps that run past the end of the game are padded as absorbing
states: zero value, zero reward, uniform policy.

Kept as a separate module specifically so Reanalyze can call it again
against refreshed values.

**`selfplay`**
Plays a complete game with MCTS over the learned model, temperature
sampling for the opening plies and greedy selection afterward, and
emits one `GameHistory`. It steps the *real* board between moves —
MuZero learns dynamics for search, not for playing.

**`eval`**
Plays the network against `minimax` as both X and O with greedy,
noise-free search. Identical in shape to the AlphaZero version, which
keeps the convergence comparison honest.

### Executables and Tools

**`train`** — self-play games append trajectories to the buffer;
sampled positions become unrolled samples; gradient steps update all
three networks jointly. Periodically checkpoints and runs a short
minimax evaluation so convergence is visible.

**`evaluate`** — scores a checkpoint against minimax.

**`play_cli`** — interactive play against a checkpoint.

**`diag_eval`** — per-move game transcript with MCTS visit
distributions, as in the sibling project.

**`latent_probe`** — the MuZero-specific diagnostic. Rolls dynamics
forward k steps from a position and compares prediction's output at
that latent against a fresh `h`+`f` pass on the real board the same
actions reach. Reports policy divergence and value error as a function
of k, which is the direct measure of whether the learned model has
actually captured the game. Also reports reward-prediction error on
terminal transitions.

## Data Flow

1. `train` runs self-play; each game uses the current network to guide
   latent MCTS and produces a `GameHistory`.
2. Trajectories land in `replay_buffer`.
3. `train` samples (trajectory, position) pairs; `targets` turns each
   into an `UnrolledSample`.
4. `network.trainStep` unrolls K steps, computes the summed loss, and
   backpropagates through time into all three networks.
5. Every N iterations (N configurable, distinct from the unroll
   length K), `train` checkpoints and evaluates against `minimax`.
6. `evaluate`, `play_cli`, and the tools independently load checkpoints.

## Training Details

**Loss.** Summed over the K+1 unroll steps:
- value: mean-squared error against the n-step bootstrap target
- policy: cross-entropy against the stored MCTS visit distribution
- reward: mean-squared error against the observed reward (steps 1..K
  only; step 0 has no incoming transition)

**Two scaling rules from the paper, both load-bearing:**
- The gradient contributed by each unroll step is scaled by 1/K, so a
  sample unrolled five steps does not outweigh one unrolled once.
- The gradient flowing into dynamics *from its latent input* is halved
  at every step. Without this, gradient magnitude compounds across the
  recurrence and latents destabilize.

**Reward on this domain.** Tic-tac-toe pays out only at the end, so
reward targets are zero everywhere except the transition into a
terminal state. The head is kept regardless: its presence is
structural to MuZero, and a network that cannot predict "this move
ends the game, and in whose favor" cannot search correctly over its
own model.

## Root-Only Legality

MuZero masks illegal actions at the root and nowhere else. Inside the
tree, all nine actions are always available.

The consequence on this domain is worth stating plainly. No training
trajectory ever contains an illegal action, so the dynamics network is
entirely unconstrained on those inputs — it will map them to whatever
latent minimizes nothing in particular, and prediction will report a
confident, meaningless value there. Search will sometimes spend
simulations on those branches. Min-max normalization and the learned
value function limit the damage but do not eliminate it.

This is precisely the price MuZero pays for not being given the rules,
and it is why more training iterations should be expected here than
AlphaZero needs. The project measures that gap rather than engineering
around it. Legal masking at every node would train faster and would
also quietly hand back the rule knowledge the algorithm exists to do
without.

## Reanalyze Readiness

Reanalyze re-runs search on stored positions with the *current*
network and replaces the stale targets recorded when the game was
played. It is out of scope here, but three decisions in this spec are
made to accommodate it later without restructuring:

1. The replay buffer stores whole trajectories, including the raw
   observation at every step — the input a fresh search needs.
2. `searchPolicies` and `searchValues` are stored as mutable
   per-trajectory arrays with accessors that permit in-place
   replacement.
3. Target construction is a pure function of a trajectory, not a step
   fused into self-play, so it can be re-invoked over refreshed values.

Adding Reanalyze should then be a new module plus a training-loop
option, touching no existing component's interface.

This is also a genuine capability asymmetry: AlphaZero cannot do this.
Its value targets are final game outcomes, which are facts about the
past and cannot be improved by a better network. MuZero's targets are
bootstrapped from its own search, so a stronger network makes old data
more valuable.

## Deviations From the Paper

- **Scalar value and reward heads** instead of the categorical
  distribution over a 601-bin support with the invertible scaling
  transform `h(x) = sign(x)(sqrt(|x|+1) - 1) + eps*x`. That machinery
  exists for Atari's large, unbounded returns. Tic-tac-toe's are
  exactly {-1, 0, +1}, and a `tanh` scalar covers the range without
  the extra layer of indirection.
- **SGD instead of Adam with the paper's learning-rate schedule.**
  Consistent with the sibling project.
- **No Reanalyze** (see above).
- **Small networks** — 64-unit hidden layers and a 32-unit latent,
  versus the paper's deep residual towers. Sized to the domain.

Each of these is recorded in `docs/muzero-vs-alphazero.md` so a reader
never mistakes a simplification for the algorithm.

## Error Handling

Matching the sibling project's minimal approach:
- `board`: illegal moves are a programming error, asserted.
- `mcts`: illegal actions inside the tree are *not* an error — they
  are the design. Only root move selection filters for legality.
- `play_cli`: human input is the one untrusted boundary; illegal or
  malformed input is rejected and re-prompted, never asserted.
- Checkpoint load failure: fail fast with a clear message, no fallback
  to random weights.

## Testing

Plain `assert`-based test executables, no framework:

- **`board`, `minimax`** — carried over from the sibling project.
- **`mlp`** — numerical gradient check per layer: perturb each weight,
  compare the finite-difference derivative against the analytical one.
  Also verifies that two `backward` calls without an intervening
  `zeroGrad` accumulate rather than overwrite.
- **`network`** — end-to-end numerical gradient check through a
  multi-step unroll, covering all three networks including dynamics
  applied repeatedly. This is where a backprop-through-time bug would
  otherwise hide silently, degrading training without ever failing.
  Also: latent min-max normalization holds outputs in [0, 1];
  save/load round-trips exactly.
- **`mcts`** — with a hand-built oracle network (so the test never
  depends on training having worked): reward-carrying backup with
  correct sign alternation; min-max normalization mapping observed Q
  onto [0, 1]; root legal masking; Dirichlet noise leaving every legal
  root action with positive probability; non-root nodes expanding all
  nine actions.
- **`targets`** — n-step values against hand-computed expectations,
  including alternating sign and the `tdSteps`-beyond-game-length case
  collapsing to the game outcome; absorbing-state padding past
  terminal.
- **`replay_buffer`** — ring eviction at capacity; sampled positions
  always in range; in-place target replacement.
- **`selfplay`** — a completed game yields a trajectory whose arrays
  have consistent lengths and whose rewards are zero except at the
  terminal transition.
- **`integration_smoke.sh`** — a short train run produces a checkpoint
  that `evaluate`, `play_cli`, and `latent_probe` all load.

## Build

CMake 3.16+, C++17, no external dependencies. One `libmz` static
library; all executables and tests link against it. Test targets
compile with `-UNDEBUG` so `assert` survives the default Release
build — the same trap the sibling project documents.

## Documentation

Three pieces, matching and extending the sibling project's approach:

1. **`docs/algorithm-explained.md`** — a guided walkthrough in the same
   voice and structure as the AlphaZero one, following the code in the
   order it runs, with inline "vs AlphaZero" callouts at each point of
   divergence. A styled `docs/algorithm-explained.html` reuses the
   existing stylesheet.
2. **`docs/muzero-vs-alphazero.md`** — the standalone comparison: a
   summary table, then a section per difference (given vs learned
   dynamics; the latent state and the loss of interpretability;
   root-only legality; the reward head; bootstrapped vs outcome value
   targets; trajectory vs position storage; the two PUCT formulas;
   Reanalyze as a structural capability), and the recorded deviations
   above.
3. **`README.md`** — build, usage, project layout, pointers to both,
   and the measured convergence comparison against the AlphaZero
   project once training has actually been run.

## Authorship

Mark Castelluccio, designed and implemented with Claude Code —
matching the sibling project.
