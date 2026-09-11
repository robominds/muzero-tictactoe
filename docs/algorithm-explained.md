# MuZero from Scratch

What happens when you take the rules away — a 9-square board, three small
networks, and a tree search that plans inside a model it invented for
itself, walked through against the actual C++ in this repository, section
by section, in the order the code actually runs.

*(A richer, illustrated version of this document — with diagrams and syntax
highlighting — is available as `docs/algorithm-explained.html`.)*

**Language:** C++17, zero dependencies · **Networks:** `h` 18 → 64 → 32 ·
`g` 41 → 64 → {32, 1} · `f` 32 → 64 → {9, 1} ·
**Result:** 1 of 3 seeds reached `draws=100 / losses=0` against perfect
play; the other two at the identical configuration reached
`draws=50 / losses=50` and `draws=0 / losses=100`

This is the companion to the AlphaZero project's own
`docs/algorithm-explained.md`, which walks the same reader through
AlphaZero on the same game. The two projects share a board, a minimax
opponent, an evaluation harness and a hand-written-backprop philosophy,
and differ in exactly one thing: whether the search is allowed to know
the rules. Every place that one difference
propagates outward gets a **vs AlphaZero** callout below, where it arises.
[`muzero-vs-alphazero.md`](muzero-vs-alphazero.md) collects them into a
standalone comparison.

## Contents

- [00 — The problem with knowing the rules](#00--the-problem-with-knowing-the-rules)
- [01 — Three functions instead of one](#01--three-functions-instead-of-one)
- [02 — The action has to become a number](#02--the-action-has-to-become-a-number)
- [03 — Searching a model you cannot see](#03--searching-a-model-you-cannot-see)
- [04 — Q values of unknown scale](#04--q-values-of-unknown-scale)
- [05 — Backup through alternating players](#05--backup-through-alternating-players)
- [06 — What the game actually pays](#06--what-the-game-actually-pays)
- [07 — Targets that improve with the network](#07--targets-that-improve-with-the-network)
- [08 — Learning through five copies of itself](#08--learning-through-five-copies-of-itself)
- [09 — The full loop](#09--the-full-loop)
- [10 — What this bought, and what it cost](#10--what-this-bought-and-what-it-cost)
- [11 — Concept map](#11--concept-map)

---

## 00 — The problem with knowing the rules

**Concept: model-based vs. model-free reinforcement learning**

An RL agent that plans has to answer a question before it can plan at all:
*if I take this action, what happens?* The function that answers it is
called a **model** of the environment. An agent that has one — and uses it
to look ahead — is doing **model-based** RL. An agent that only learns
which actions turned out well, without ever asking what they do, is doing
**model-free** RL.

AlphaZero is model-based in the easiest possible way: somebody handed it a
perfect model. Its search calls `board.applyMove(m)` and gets the exact
resulting position, calls `board.isTerminal()` and gets the exact truth
about whether the game is over, calls `board.legalMoves()` and gets exactly
the moves that exist. None of that is learned. None of it can be wrong.

That is a completely reasonable thing to rely on for tic-tac-toe, and for
chess, and for Go. It is useless the moment the environment is a physical
robot, a market, a protein, or an Atari emulator you are only allowed to
observe through its pixel output. In all of those, "apply this action and
tell me the resulting state" is precisely the thing you do not have.

MuZero's answer is to **learn the model along with everything else** — and,
crucially, to learn only the parts of it that planning actually needs. It
gives up:

- knowing which moves are legal below the search root,
- knowing when the game has ended,
- being able to look at a searched position and say what it is,
- and, as this project measured, a great deal of reliability.

And it buys: an algorithm whose search machinery does not care whether a
simulator exists. The same code that plays tic-tac-toe here is, structurally,
the code that plays Atari from pixels.

So: **if you cannot simulate the game, what exactly do you need to predict
in order to search?** Not the next board — nothing in a tree search ever
reads a board for its own sake. Search only ever asks a position three
questions: *which moves look good here* (a policy), *how good is this for
me* (a value), and *what did I collect getting here* (a reward). Predict
those three, consistently, over some internal representation of your own
choosing, and the search will run. That is the whole idea, and the next
eleven sections are what it costs.

> **Why this matters beyond tic-tac-toe.** "Learn only the parts of the
> model that the downstream computation consumes" is one of the most
> transferable ideas in modern RL. The classical alternative — learn to
> predict the next observation, then plan against your predictions — spends
> almost all its capacity on detail that no decision depends on. A model
> trained end-to-end through the planner is under no obligation to be a
> simulator; it is only obliged to be *useful to a planner*. That is a much
> weaker requirement, and weaker requirements are learnable from less data.

---

## 01 — Three functions instead of one

**Concept: latent dynamics models**

AlphaZero has one network: board in, policy and value out. MuZero has
three, and the split is the whole architecture.

```
     real board (18 floats)
            │
            ▼
     ┌─────────────┐
     │  h : repr.  │   18 → 64 → 32      called exactly once per search,
     └─────────────┘                     at the root, on a real observation
            │
            ▼
        latent s⁰ (32) ──────────────┐
            │                        │
            │  ┌──────────────┐      │   ┌──────────────┐
            └─▶│ g : dynamics │      └──▶│ f : predict  │  32 → 64 → {9, 1}
       action ▶│ 41 → 64 →    │          └──────────────┘
        (9)    │   {32, 1}    │                 │
               └──────────────┘                 ▼
                    │    │                  policy (9), value (1)
                    │    └──▶ reward (1)
                    ▼
                latent s¹ (32) ──▶ g ──▶ s² ──▶ g ──▶ s³ ⋯
```

- **`h`, the representation function.** Real observation → latent. Runs
  once, at the search root. This is the only place in the entire system
  where a real board is converted into something the networks can read.
- **`g`, the dynamics function.** (latent, action) → (next latent, reward).
  This replaces `Board::applyMove`. It is a neural network, it is wrong a
  lot, and section 03 and section 10 are both about how wrong.
- **`f`, the prediction function.** Latent → (policy, value). This is
  exactly AlphaZero's two-headed network, except that it reads a latent
  instead of a board.

| function | input | hidden | output |
|---|---|---|---|
| `h` representation | observation (18) | 64, ReLU | latent (32), min-max normalized |
| `g` dynamics | latent (32) + action one-hot (9) = 41 | 64, ReLU | latent (32), min-max normalized; reward (1), `tanh` |
| `f` prediction | latent (32) | 64, ReLU | policy (9), softmax; value (1), `tanh` |

Now the negative fact that surprises everybody, and which is stated as a
design constraint in the header rather than left to be inferred:

**There is no decoder.** Nothing in this codebase can take a latent and
say which board it is. `MuZeroNetwork` has `h`, `g` and `f` and nothing
that runs backwards. If you stop the search mid-simulation and look at the
node it is standing on, you get 32 floats in `[0, 1]` and no way to say
what position they mean.

```cpp
// There is deliberately NO decoder from latent back to a board. Nothing
// here can look at a latent and say which position it is. A latent only
// has to support good policy, value and reward predictions -- it is never
// asked to reconstruct the observation. That is the constraint that
// separates MuZero from AlphaZero, which always has the real board.
```
*include/mz/network.hpp:17 — `MuZeroNetwork`, class comment*

Why is that allowed? Because of the answer in section 00: the latent is
never asked to *be* a board. It is only ever asked to support three
predictions. The loss function never mentions a board below step 0, so the
network is never penalized for representing a position unrecognizably — and
so it does, in whatever coordinates make the three predictions easiest. A
latent is a set of beliefs about the future of a position, not a picture of
it.

Here are the two entry points. `initialInference` is `h` then `f`:

```cpp
MuZeroNetwork::InitialInference MuZeroNetwork::initialInference(
    const std::array<float, kObservationSize>& observation, Workspace& ws) const {
    ws.observation.assign(observation.begin(), observation.end());
    hFc1_.forwardInto(ws.observation, ws.trunkPre);
    reluInto(ws.trunkPre, ws.trunk);
    hFc2_.forwardInto(ws.trunk, ws.latentPre);

    InitialInference out;
    // The latent is real output, not scratch -- the caller keeps it.
    minMaxNormalizeInto(ws.latentPre, out.latent);

    fFc1_.forwardInto(out.latent, ws.predPre);
    reluInto(ws.predPre, ws.pred);
    fPolicy_.forwardInto(ws.pred, ws.logits);
    out.policy = softmax9(ws.logits);
    fValue_.forwardInto(ws.pred, ws.scalar);
    out.value = std::tanh(ws.scalar[0]);
    return out;
}
```
*src/network.cpp:52 — `MuZeroNetwork::initialInference` (the Workspace overload)*

(There is also a one-argument overload, `initialInference(observation)`, a thin wrapper that builds a
throwaway `Workspace` and forwards to this one — for callers, like the tests, that do not care about
reuse. The workspace is discussed below.)

And `recurrentInference` is `g` then `f` — the one that runs everywhere
else, and which never sees an observation at all:

```cpp
MuZeroNetwork::RecurrentInference MuZeroNetwork::recurrentInference(const std::vector<float>& latent,
                                                                    int action, Workspace& ws) const {
    fillDynamicsInput(latent, action, ws.dynamicsInput);
    gFc1_.forwardInto(ws.dynamicsInput, ws.trunkPre);
    reluInto(ws.trunkPre, ws.trunk);
    gFc2_.forwardInto(ws.trunk, ws.latentPre);

    RecurrentInference out;
    minMaxNormalizeInto(ws.latentPre, out.latent);
    gReward_.forwardInto(ws.trunk, ws.scalar);
    out.reward = std::tanh(ws.scalar[0]);

    fFc1_.forwardInto(out.latent, ws.predPre);
    reluInto(ws.predPre, ws.pred);
    fPolicy_.forwardInto(ws.pred, ws.logits);
    out.policy = softmax9(ws.logits);
    fValue_.forwardInto(ws.pred, ws.scalar);
    out.value = std::tanh(ws.scalar[0]);
    return out;
}
```
*src/network.cpp:78 — `MuZeroNetwork::recurrentInference` (the Workspace overload)*

Notice that `f` — `fFc1_`, `reluInto`, `fPolicy_`, `fValue_` — runs
identically in both, into the same `ws.predPre`/`ws.pred` scratch. One
prediction network, shared between "a real position I just looked at" and
"a position I imagined four plies deep." That sharing is what forces the
latents produced by `g` to live in the same space as the latents produced
by `h`: they both have to be readable by the same `f`.

> **vs AlphaZero.** AlphaZero has one network and one function signature:
> board in, (policy, value) out. Everything about the transition between
> positions — legality, terminality, the resulting board — is supplied by
> `Board`, which is ordinary non-learned code and is never wrong. MuZero
> replaces that supplied transition with `g`, a 41 → 64 → {32, 1} MLP that
> is wrong constantly. Everything else in this document follows from that
> substitution. See [`muzero-vs-alphazero.md`](muzero-vs-alphazero.md) for
> the full accounting.

---

## 02 — The action has to become a number

**Concept: action encoding**

In AlphaZero, "apply action 4" is `board.applyMove(4)` — the integer indexes
a cell, the cell gets a stone, done. In MuZero there is no board and no
cell. The action has to reach the dynamics network as *input*, alongside
the state, and the only thing a `Dense` layer accepts is a vector of
floats. So the action becomes nine floats:

```cpp
void MuZeroNetwork::fillDynamicsInput(const std::vector<float>& latent, int action,
                                      std::vector<float>& out) {
    assert(static_cast<int>(latent.size()) == kLatentSize);
    assert(action >= 0 && action < kActionSize);
    // assign, not resize: the one-hot tail must be zeroed even when this
    // buffer is being reused from a previous action.
    out.assign(kLatentSize + kActionSize, 0.0f);
    std::copy(latent.begin(), latent.end(), out.begin());
    out[kLatentSize + action] = 1.0f;
}
```
*src/network.cpp:31 — `MuZeroNetwork::fillDynamicsInput`*

**Example.** Latent `s = [s₀ … s₃₁]`, action 4 (the centre square):

| input index | 0 … 31 | 32 | 33 | 34 | 35 | **36** | 37 | 38 | 39 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|
| value | s₀ … s₃₁ | 0 | 0 | 0 | 0 | **1** | 0 | 0 | 0 | 0 |

That is the entire mechanism. One-hot rather than the raw integer 4,
because a `Dense` layer is linear in its input and would otherwise be told
that action 8 is "twice as much action" as action 4 — an ordering that
means nothing on a 3×3 grid. Nine independent input columns let the network
learn nine unrelated things.

And note what is *absent*: no legality check, and no way to express "that
move is impossible." `fillDynamicsInput` will cheerfully encode action 4 on
a latent whose real board already has a stone on square 4. `g` will
cheerfully produce a next latent for it. That is not an oversight — it is
section 03.

### Keeping the recurrence from drifting

`g` is applied to its own output, over and over, down a search path. That
is a recurrence, and recurrences are where magnitudes go to explode. MuZero
(appendix G) handles it by scaling every latent into `[0, 1]` across its
own elements, immediately on production:

```cpp
void minMaxNormalizeInto(const std::vector<float>& z, std::vector<float>& out) {
    assert(!z.empty());
    float lo = *std::min_element(z.begin(), z.end());
    float hi = *std::max_element(z.begin(), z.end());
    float denom = std::max(hi - lo, kMinMaxFloor);
    out.resize(z.size());
    for (size_t i = 0; i < z.size(); ++i) out[i] = (z[i] - lo) / denom;
}
```
*src/mlp.cpp:154 — `minMaxNormalizeInto`*

(`minMaxNormalize`, the allocating form used by the tests and by the
single-argument inference overloads, is a two-line wrapper around this.)

Both `h` (line 91 of `src/network.cpp`) and `g` (line 116) push their output
through it, so *every* latent in the system, no matter how it was produced
or how deep, has minimum exactly 0 and maximum exactly 1. The `kMinMaxFloor`
of `1e-5` is there for the degenerate case where every element is equal —
which really happens, on the empty board, where an all-zero observation
drives a bias-only forward pass through every layer.

`test_deep_unroll_stays_bounded` is the test that holds this in place. It
applies `g` **fifty** times — far deeper than any search will ever go —
and asserts on every step that the latent is finite and inside
`[-1e-6, 1 + 1e-6]`, and that value and reward stay finite:

```cpp
void test_deep_unroll_stays_bounded() {
    // Apply dynamics far deeper than search ever will. Min-max
    // normalization must keep latents in range no matter how many times
    // the recurrence is applied.
    MuZeroNetwork network(17);
    Board board;
    auto state = network.initialInference(board.encode());
    std::vector<float> latent = state.latent;
    for (int step = 0; step < 50; ++step) {
        auto next = network.recurrentInference(latent, step % 9);
```
*tests/test_network.cpp:139 — `test_deep_unroll_stays_bounded` (assertions elided)*

Bounded is not the same as correct — section 10 has the measurement showing
how far these latents drift from the real game even when they stay in
range. But unbounded would be fatal rather than merely inaccurate, and this
is the one line that prevents it.

> **vs AlphaZero.** AlphaZero's state representation cannot drift, because
> it is not produced by anything — it is the board. There is no equivalent
> of `minMaxNormalize` in the sibling project because there is no
> recurrence to stabilize.

---

## 03 — Searching a model you cannot see

**Concept: MCTS over learned state**

The search is PUCT-based Monte Carlo Tree Search, structurally the same as
AlphaZero's: repeatedly walk from the root down to an unexpanded node,
expand it, and back its value up the path. What changes is what "walking
down" means.

At the root — and only at the root — the real world enters:

```cpp
MCTSResult MCTS::run(const Board& board, float temperature) {
    assert(!board.isTerminal());
    const std::vector<int> legalActions = board.legalMoves();

    nodeCount_ = 0;
    const int rootIndex = acquireNode();
    MuZeroNetwork::InitialInference initial = network_.initialInference(board.encode(), workspace_);
    nodes_[rootIndex].latent = initial.latent;

    // The one and only place legality enters the search.
    std::array<float, 9> rootPriors = maskAndRenormalize(initial.policy, legalActions);
    if (config_.addRootNoise) {
        mixDirichletNoise(rootPriors, legalActions, rng_, config_.dirichletAlpha,
                          config_.dirichletEpsilon);
    }
    for (int a : legalActions) {
        int child = acquireNode();
        nodes_[child].prior = rootPriors[a];
        nodes_[rootIndex].children[a] = child;
    }
    nodes_[rootIndex].expanded = true;
```
*src/mcts.cpp:136 — `MCTS::run` (root setup)*

Two real-world facts are consumed here and nowhere else: the encoded
observation, and `legalMoves()`. The root gets children only for legal
actions. Everything below is latents.

Nodes here are not individually allocated. `nodes_` is an arena — a
`std::vector<Node>` reused across searches — and a child is addressed by
its integer index into it rather than owned through a pointer. `acquireNode`
hands back a reset slot, growing the arena only the first time a search
needs more of it than the last one did; in steady state a search touches
the allocator zero times. `workspace_` is the same `MuZeroNetwork::Workspace`
from section 01, held once per `MCTS` and passed into every inference this
search makes.

Expansion below the root is where the difference becomes visible:

```cpp
        const int parentIndex = path_[path_.size() - 2];
        MuZeroNetwork::RecurrentInference step =
            network_.recurrentInference(nodes_[parentIndex].latent, lastAction, workspace_);

        nodes_[nodeIndex].latent = std::move(step.latent);
        nodes_[nodeIndex].stats.reward = step.reward;
        // Below the root, every one of the nine actions gets a child.
        // There is no legality oracle here and no terminal detection: the
        // search does not know the rules, and has to learn from the value
        // and reward heads that some of these branches are worthless.
        for (int a = 0; a < 9; ++a) {
            int child = acquireNode();
            nodes_[child].prior = step.policy[a];
            // Re-index: acquireNode may have grown the arena, so this must
            // not be hoisted into a Node& held across the loop.
            nodes_[nodeIndex].children[a] = child;
        }
        nodes_[nodeIndex].expanded = true;
        ++nodesExpanded;
        maxDepth = std::max(maxDepth, static_cast<int>(path_.size()) - 1);
```
*src/mcts.cpp:174 — `MCTS::run` (expansion)*

`for (int a = 0; a < 9; ++a)`. Not `for (int a : legalActions)`. Every node
below the root gets all nine children, including the ones that would put a
second stone on an occupied square, and including nodes whose real
counterpart is a finished game. There is no `isTerminal()` anywhere in this
file. The tree does not know when the game ends, because the tree does not
know what a game is.

### The illegal-branch problem, honestly

This is not free, and the project measured exactly what it costs. Running
`tools/diag_eval` at 400 simulations reports search tree depths of **18–19
on a nine-square board** (`docs/results.md`). A real tic-tac-toe game cannot
be more than 9 plies deep. So by the time a typical simulation terminates,
it is standing at a depth where no legal game could still be running:
imagined moves layered on imagined moves, in positions the dynamics network
invented and the real rules would never reach. As `docs/results.md` puts
it, this is not "search sometimes explores an illegal branch," it is "the
median simulation is illegal by the time it terminates."

The only thing that pushes back is learning. Walking into an impossible
branch has to become unattractive because the value and reward heads
eventually say so — the same way everything else here is learned. On this
domain, over 23 training runs, that pushback was never strong enough to
stop the tree from going twice as deep as the game is long.

`test_tree_grows_below_a_root_with_only_one_legal_move` is the test that
proves the tree really is unmasked, and it is a nice piece of test design
because it constructs the one position where a rules-aware search
*physically cannot* do what this one does:

```cpp
    Board board;
    int moves[] = {0, 1, 2, 4, 3, 5, 7, 6};   // eight plies, square 8 open
    for (int m : moves) board = board.applyMove(m);
    assert(board.legalMoves().size() == 1);
    assert(!board.isTerminal());
```
*tests/test_mcts.cpp:163 — `test_tree_grows_below_a_root_with_only_one_legal_move`*

One legal move, and playing it ends the game. A search that knew the rules
would have exactly one child at the root and nothing at all below it: depth
1, forever. The test asserts `result.maxDepth >= 2`. It passes.

> **vs AlphaZero.** AlphaZero applies legality at every node, because
> `Board` is there at every node, and it stops at terminal positions with
> the exact outcome instead of a network estimate — which is why its search
> can find a forced win with a handful of simulations. MuZero's tree has
> neither. That single decision is the biggest source of wasted compute in
> this project and, per `docs/results.md`, plausibly the biggest source of
> its unreliability. See
> [`muzero-vs-alphazero.md`](muzero-vs-alphazero.md) on root-only legality;
> the spec's "Root-Only Legality" section argues why masking below the root
> would quietly hand back the rule knowledge the algorithm exists to do
> without.

---

## 04 — Q values of unknown scale

**Concept: value normalization in search**

PUCT compares two things that have to be on comparable scales: the
accumulated evidence about a move (`Q`) and an exploration bonus derived
from a prior probability. In AlphaZero that comparison is easy, because
`Q` is a mean of `tanh` outputs and terminal outcomes — always in
`[-1, 1]`, always directly comparable with a prior in `[0, 1]`.

MuZero's `Q` is not that. It is reward plus a discounted, negated value:

```cpp
float edgeQ(float childReward, float childValue, float discount) {
    return childReward + discount * -childValue;
}

float explorationTerm(float prior, int parentVisits, int childVisits) {
    float pbC = std::log((static_cast<float>(parentVisits) + kPbCBase + 1.0f) / kPbCBase) + kPbCInit;
    return prior * std::sqrt(static_cast<float>(parentVisits)) / (1.0f + static_cast<float>(childVisits)) * pbC;
}
```
*src/mcts.cpp:28 — `edgeQ` and `explorationTerm`*

The reward comes out of a learned head. Early in training it can be
anything the head happens to emit; in a general MuZero domain, rewards are
not even bounded. So the scale of `Q` is not known in advance, which means
a fixed `c_puct` cannot be tuned against it. MuZero's fix is to normalize
`Q` against the range of `Q` values *this search tree has actually seen*:

```cpp
float MinMaxStats::normalize(float value) const {
    if (!seeded_ || max_ <= min_) return value;
    return (value - min_) / (max_ - min_);
}
```
*src/mcts.cpp:19 — `MinMaxStats::normalize`*

A running min and max, per search, updated on every backed-up edge (section
05), mapping `Q` into `[0, 1]` regardless of what units it started in. And
then selection:

```cpp
    for (int a = 0; a < 9; ++a) {
        if (parent.children[a] < 0) continue;   // root only, for illegal moves
        const Node* child = &nodes_[parent.children[a]];
        // An unvisited child scores 0 on the Q side, matching MuZero's
        // pseudocode: after normalization that is "as bad as the worst
        // thing seen so far", so exploration has to come from the prior.
        float q = 0.0f;
        if (child->stats.visitCount > 0) {
            q = stats.normalize(edgeQ(child->stats.reward, nodeValue(child->stats), kDiscount));
        }
        float score = q + explorationTerm(child->prior, parent.stats.visitCount, child->stats.visitCount);
        if (score > bestScore) {
            bestScore = score;
            bestAction = a;
        }
    }
```
*src/mcts.cpp:116 — `MCTS::selectChild`*

`parent.children[a]` is an arena index now, not a `unique_ptr<Node>`; `-1`
plays the role `nullptr` used to, and `&nodes_[...]` takes a pointer into
the arena for the rest of the function's read-only use.

The two formulas, side by side:

```
AlphaZero:  score(a) = Q(s,a) + c_puct · P(a|s) · √(Σ_b N(s,b)) / (1 + N(s,a))

MuZero:     score(a) = normalize(Q(s,a))
                     + P(a|s) · √N(s) / (1 + N(s,a)) · [ c₁ + log((N(s) + c₂ + 1) / c₂) ]
```

The exploration half is identical in shape. The differences are the
`normalize` wrapper — section-04's whole subject — and the replacement of
AlphaZero's constant `c_puct` with `c₁ + log((N + c₂ + 1)/c₂)`, a term that
grows slowly with the parent's visit count so that heavily-visited nodes
keep exploring a little longer.

**And here is the honest note about that term at this scale.** `c₂ = 19652`
(`kPbCBase`), the paper's value, tuned for searches of hundreds of
thousands of simulations. At tic-tac-toe's budgets it barely moves:

| parent visits N | log((N + 19652 + 1)/19652) | full factor c₁ + log(⋯) |
|---|---|---|
| 1 | 0.000102 | 1.250102 |
| 10 | 0.000560 | 1.250560 |
| 100 | 0.005126 | 1.255126 |
| 300 | 0.015200 | 1.265200 |

Over an entire 100-simulation search, the log term contributes a total
swing of about 0.005 on a factor of 1.25 — four tenths of one percent.
Functionally this is AlphaZero's fixed `c_puct ≈ 1.25`. The header comment
in `include/mz/mcts.hpp` says as much: it is here "because the mechanism,
not its magnitude, is the point." Keeping it means this code is the real
formula rather than a simplification a reader would then have to un-learn;
pretending it does work here would be a lie.

> **vs AlphaZero.** `MinMaxStats` has no counterpart in the sibling
> project, and it exists for exactly one reason: MuZero's `Q` includes a
> learned reward, so nobody knows its scale in advance. This is the
> smallest of the divergences and the most purely mechanical — it is what
> you must add to PUCT once values stop being guaranteed to live in
> `[-1, 1]`.

---

## 05 — Backup through alternating players

**Concept: credit assignment with intermediate rewards**

A simulation ends at a freshly expanded node with a value estimate from
`f`. That estimate has to travel back up the path it came down, updating
every node on the way. In a two-player zero-sum game it also has to flip
sign at every ply — and in MuZero it has to pick up each node's predicted
reward as it passes.

```cpp
void backupPath(const std::vector<BackupNode*>& path, float leafValue, float discount,
                MinMaxStats& stats) {
    assert(!path.empty());
    // `value` is always in the perspective of the player to move at the
    // node currently being updated.
    float value = leafValue;
    for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
        BackupNode* node = path[i];
        node->valueSum += value;
        node->visitCount += 1;
        if (i > 0) {
            // The Q of the edge that led here, which is exactly the
            // quantity selectChild compares.
            stats.update(edgeQ(node->reward, nodeValue(*node), discount));
        }
        // Hand it up to the parent: collect this node's reward, flip sign.
        value = node->reward + discount * -value;
    }
}
```
*src/mcts.cpp:37 — `backupPath`*

The single line that carries the whole algorithm is
`value = node->reward + discount * -value;`. Read it as an instruction to
the parent's player: *you collect the reward this transition paid you, and
then you inherit the negation of how good the position is for your
opponent.*

### Two perspective conventions, stated once

Every sign in this project depends on these two sentences, and they are
written into the headers so they cannot drift apart:

- **A value is from the perspective of the player to move at that node.**
  (`include/mz/network.hpp:24`, `include/mz/game_history.hpp:18`.)
- **A reward is from the perspective of the player who took the action that
  earned it.** (`include/mz/network.hpp:25`, `include/mz/mcts.hpp:44`.)

They must agree across the network's heads, the search backup, the
self-play reward assignment, and the n-step value target — four independent
places. A disagreement between any two would compile, run, and quietly
train the network to prefer losing.

**Worked example**, the exact fixture from
`test_backup_alternates_sign_and_carries_reward`. Three nodes, root → A →
B. The move into B paid its mover `+0.5`. The leaf evaluates to `+0.8`
for whoever moves at B. Discount 1.0.

| step | node | reward on the edge in | value credited here | how it was computed |
|---|---|---|---|---|
| leaf | **B** | +0.5 | **+0.80** | the leaf estimate, as-is |
| ↑ | **A** | 0.0 | **−0.30** | `0.5 + 1.0 × −0.8` |
| ↑ | **root** | 0.0 | **+0.30** | `0.0 + 1.0 × −(−0.3)` |

Walk it in words. B's player sees a position worth +0.8 to them. A's player
took the move into B, which paid them +0.5 — and then handed their opponent
a position worth +0.8 to that opponent, i.e. −0.8 to A. So A's total is
+0.5 − 0.8 = −0.3. The root player, whose move led to A, collected nothing
and handed over a position worth −0.3 to A, i.e. +0.3 to them. The test
asserts exactly `0.8`, `-0.3`, `0.3`.

Note the `if (i > 0)` guard: the root's own edge-`Q` is never recorded into
`MinMaxStats`, because the root has no incoming edge. The companion test
`test_backup_records_edge_q_but_not_the_root` pins that.

> **Why this matters beyond tic-tac-toe.** Intermediate rewards are the
> normal case in RL — an Atari game pays points continuously, a robot pays
> energy costs at every step. AlphaZero's backup has no reward term at all
> because board games pay exactly once, at the end. MuZero's backup is
> written for the general case and then handed a domain where the reward
> happens to be zero almost everywhere. That the *same* line of code covers
> both is the point.

---

## 06 — What the game actually pays

**Concept: reward as distinct from value**

**Reward** is what the environment pays you for one transition. **Value** is
what you expect to collect from here to the end. They are different
quantities and MuZero predicts both, with separate heads.

On tic-tac-toe that distinction is almost degenerate, because the game pays
nothing until it is over:

```cpp
        if (board.isTerminal()) {
            // Reward is in the perspective of the player who just moved.
            Outcome outcome = board.outcome();
            float reward = 0.0f;
            if (outcome != Outcome::Draw) {
                bool moverWon = (outcome == Outcome::XWins && mover == Cell::X) ||
                                (outcome == Outcome::OWins && mover == Cell::O);
                reward = moverWon ? 1.0f : -1.0f;
            }
            game.rewards.back() = reward;
        }
```
*src/selfplay.cpp:40 — `playSelfPlayGame`*

Every entry of `game.rewards` was pushed as `0.0f` a few lines earlier
(`src/selfplay.cpp:30`); only the last one is ever overwritten, and only
when the final move actually won or lost. In a five-ply game the reward
targets are `[0, 0, 0, 0, +1]`. In a nine-ply draw they are all zero. The
reward head spends the overwhelming majority of its training predicting
exactly zero.

So why keep it? Because the reward head is the model's *only* way to know
that games end and who won. Strip it out and `g` becomes a function that
maps latents to latents forever, with no notion of an outcome anywhere in
it — and then search over that model has nothing to search *for*. A model
that cannot tell when the game is over, and in whose favor, cannot be
planned in at all. The value head alone will not do it either: value is a
prediction *about* future rewards, so a value head with nothing underneath
it is predicting the output of a function that does not exist.

**How well does it work here?** Not well. `tools/latent_probe`, run against
the one converged checkpoint over 2000 trajectories, reports:

```
Terminal reward prediction: mean |error| = 0.7682 over 1153 transitions
```
*`docs/results.md`, `latent_probe` section*

Rewards live in `{−1, 0, +1}`. A mean absolute error of 0.77 on that set is
very large — this is a model that frequently does not know a game just
ended, or thinks one ended that did not. And this is the checkpoint that
plays perfect defensive tic-tac-toe (`draws=100 losses=0`). Both things are
true at once, and section 10 comes back to why.

> **vs AlphaZero.** There is no reward head in the sibling project and
> nowhere it could go. `Board::outcome()` answers the question exactly,
> for free, at every node. The entire reward apparatus here — a head, a
> loss term, a target array, a perspective convention, a diagnostic — is
> the price of not having that call available.

---

## 07 — Targets that improve with the network

**Concept: bootstrapping and temporal-difference learning**

This is the deepest difference between the two projects, and the one with
consequences that reach furthest past tic-tac-toe.

AlphaZero's value target is the final game result, `z`, stamped onto every
position in the game once it ends. It is a **Monte Carlo** target: an
unbiased sample of the position's true value, obtained by actually playing
to the end. It is also a **fact about the past**. A better network cannot
improve it, because there is nothing about it to improve — the game really
did end that way.

MuZero's value target is an **n-step bootstrapped return**: sum up the
actual rewards over the next `n` plies, then add the network's *own search
value* at ply `t + n` as a stand-in for everything after that.

```
z_t = Σ_{i ∈ [t, t+n)} (−1)^(i−t) · γ^(i−t) · reward_i
      + (−1)ⁿ · γⁿ · searchValue_{t+n}
```

```cpp
float bootstrappedValue(const GameHistory& game, int t, const TargetConfig& config) {
    const int length = static_cast<int>(game.length());
    if (t >= length) return 0.0f;   // absorbing state past the end of the game

    float value = 0.0f;

    const int bootstrapIndex = t + config.tdSteps;
    if (bootstrapIndex < length) {
        float sign = (config.tdSteps % 2 == 0) ? 1.0f : -1.0f;
        value += sign * std::pow(config.discount, static_cast<float>(config.tdSteps)) *
                 game.searchValues[bootstrapIndex];
    }

    const int last = bootstrapIndex < length ? bootstrapIndex : length;
    for (int i = t; i < last; ++i) {
        float sign = ((i - t) % 2 == 0) ? 1.0f : -1.0f;
        value += sign * std::pow(config.discount, static_cast<float>(i - t)) * game.rewards[i];
    }

    return value;
}
```
*src/targets.cpp:18 — `bootstrappedValue`*

**Bootstrapping** means estimating a value partly from another estimate
rather than entirely from observed outcomes. It is the defining move of
temporal-difference learning, and its trade is the classic one: lower
variance than Monte Carlo (you stop accumulating the randomness of the
remaining game), at the cost of bias (your stand-in estimate is wrong until
the network is good).

The alternating sign is the two-player part. `rewards[i]` and
`searchValues[i]` are each stored in the perspective of whoever was to move
at their own ply, and the player to move alternates, so every ply of
separation from `t` flips the sign — the same convention as the search
backup in section 05, applied along the trajectory instead of up the tree.

**Worked example**, from the `test_short_td_bootstraps_from_stored_search_value`
fixture. A five-ply game that X wins on the last move:

| ply t | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| player to move | X | O | X | O | X |
| `searchValues[t]` | 0.10 | −0.20 | 0.30 | −0.40 | 0.50 |
| `rewards[t]` | 0 | 0 | 0 | 0 | **+1** (to X, who just moved) |

With `tdSteps = 2`, target for ply 0:

| term | value |
|---|---|
| rewards over `[0, 2)` | `0 + 0` = 0 |
| bootstrap: `(−1)² · 1.0² · searchValues[2]` | `+0.30` |
| **`targetValues[0]`** | **`0.30`** |

The gap is 2 plies — even — so ply 2's value is already in X's perspective
and enters unnegated. From ply 1 the bootstrap lands on ply 3, also an even
gap, giving `−0.40`. And `test_odd_td_gap_flips_the_sign` checks the other
parity: `tdSteps = 1` from ply 0 bootstraps on ply 1, whose stored `−0.20`
is O's opinion, so it enters X's target as `+0.20`.

### The default turns all of this off

`TargetConfig::tdSteps` defaults to **32**, and the longest possible
tic-tac-toe game is 9 plies. So `bootstrapIndex < length` is false for
every position in every game: the bootstrap term never fires, the loop runs
to the end of the game, and the expression collapses to the sign-flipped
sum of the rewards — which, since only the last reward is nonzero, is
exactly the game outcome in each position's own perspective.
`test_long_td_collapses_to_game_outcome` asserts precisely that: targets
`+1, −1, +1, −1` from ply 0 of an X win.

In other words, **as shipped, this project's value target is AlphaZero's
value target.** That is a deliberate choice — it is what board-game MuZero
uses — but it means the bootstrapping machinery above is present, tested,
and dormant.

What changes when you lower it: the value head stops being trained toward
"what happened" and starts being trained toward "what I currently think
will happen," refreshed every time the trajectory is sampled. Lower
variance, and a target that improves as the network improves. It also
introduces a failure mode Monte Carlo does not have — if the search values
are systematically wrong, the target teaches the network to agree with its
own mistake.

Was it measured here? **No**, and `docs/results.md` is explicit that this is
a real gap rather than a judgment: every one of the 23 training runs used
`tdSteps = 32`. The results document also sharpens the case for trying it,
from the probe finding that the value head is *already* correct (right
signs, sensible magnitudes in both directions) under the outcome-target
regime — so lowering `tdSteps` would not be repairing a broken value head,
it would be asking whether a denser, earlier value signal changes what the
*policy* head learns, since both heads share the trunk of `f`.

> **vs AlphaZero.** This is the deepest divergence in the project, and it
> is worth separating what it costs from what it enables. It costs
> complexity and a tuning knob with no measured setting. What it enables is
> structural: because MuZero's target is the network's own opinion rather
> than a historical fact, **old data gets better as the network gets
> better**. That is the precondition for Reanalyze — periodically re-running
> search over stored trajectories with the current network and overwriting
> their targets in place. AlphaZero cannot do this at any price: its
> targets are final outcomes, and a stronger network cannot make a finished
> game have ended differently. The spec's "Reanalyze Readiness" section
> lists the three decisions here (whole-trajectory storage, mutable search
> targets, pure target construction) that leave room for it;
> `ReplayBuffer::replaceSearchTargets` is the hook, already written and
> currently unused. See
> [`muzero-vs-alphazero.md`](muzero-vs-alphazero.md) for the full treatment.

---

## 08 — Learning through five copies of itself

**Concept: backpropagation through time**

One training sample is not one position. It is an observation plus `K = 5`
actions, and the loss is evaluated at all six unroll steps at once. The
forward pass runs `h` once, then `g` five times, then `f` six times:

```
observation
     │ h
     ▼
   s⁰ ──g(a⁰)──▶ s¹ ──g(a¹)──▶ s² ──g(a²)──▶ s³ ──g(a³)──▶ s⁴ ──g(a⁴)──▶ s⁵
     │             │  r¹         │  r²         │  r³         │  r⁴         │  r⁵
     │ f           │ f           │ f           │ f           │ f           │ f
     ▼             ▼             ▼             ▼             ▼             ▼
  π⁰ v⁰         π¹ v¹         π² v²         π³ v³         π⁴ v⁴         π⁵ v⁵
     └─ compared against targetPolicies[k], targetValues[k], targetRewards[k] ─┘
```

```cpp
        for (int k = 0; k < K; ++k) {
            fillDynamicsInput(ws.latent[k], sample.actions[k], ws.dynamicsInput[k]);
            gFc1_.forwardInto(ws.dynamicsInput[k], ws.dynamicsPre[k]);
            reluInto(ws.dynamicsPre[k], ws.dynamicsAct[k]);
            gFc2_.forwardInto(ws.dynamicsAct[k], ws.latentPre[k + 1]);
            minMaxNormalizeInto(ws.latentPre[k + 1], ws.latent[k + 1]);
            gReward_.forwardInto(ws.dynamicsAct[k], ws.scalarGrad);
            ws.reward[k + 1] = std::tanh(ws.scalarGrad[0]);
        }
```
*src/network.cpp:208 — `MuZeroNetwork::trainStep` (forward unroll)*

Every intermediate is kept — `ws.latentPre` as well as `ws.latent`, the
pre-activations as well as the activations — because the backward pass
needs each of them again. `ws` here is a `TrainScratch`, `trainStep`'s own
reusable buffer (below); unrolling one sample used to allocate on the order
of sixty vectors, and settling their sizes once, on the first sample, is
what removed that traffic. This is **backpropagation through time**: the
same three networks appear many times in one computation graph, and their
gradients are the *sum* over every appearance.

Two properties of `Dense` exist solely for this, and are documented as
load-bearing in `include/mz/mlp.hpp:13`:

- `backward()` takes its input **explicitly** rather than caching it during
  `forward()`. `g` is applied five times on five different latents; a single
  cached input would silently use the wrong one.
- `backward()` **accumulates** into the gradient buffers instead of
  assigning. Those five applications must sum. `zeroGrad()` is called once
  per training step, never once per unroll step.

The reverse pass is one loop from `k = K` down to `0`:

```cpp
        for (int k = 0; k <= K; ++k) ws.dLatent[k].assign(kLatentSize, 0.0f);

        for (int k = K; k >= 0; --k) {
            float scale = lossScale(k);

            // prediction head at step k
            ws.scalarGrad.assign(1, scale * 2.0f * (ws.value[k] - sample.targetValues[k]) *
                                        (1.0f - ws.value[k] * ws.value[k]));
            ws.dPolicyLogits.resize(kActionSize);
            for (int a = 0; a < kActionSize; ++a) {
                // d(cross-entropy o softmax)/d(logit) = p - target
                ws.dPolicyLogits[a] = scale * (ws.policy[k][a] - sample.targetPolicies[k][a]);
            }

            fValue_.backwardInto(ws.predictionAct[k], ws.scalarGrad, ws.dPredictionHidden);
            fPolicy_.backwardInto(ws.predictionAct[k], ws.dPolicyLogits, ws.dFromPolicy);
            for (int i = 0; i < kHiddenSize; ++i) ws.dPredictionHidden[i] += ws.dFromPolicy[i];

            reluBackwardInto(ws.predictionPre[k], ws.dPredictionHidden, ws.dPredictionPre);
            fFc1_.backwardInto(ws.latent[k], ws.dPredictionPre, ws.dFromPrediction);
            for (int i = 0; i < kLatentSize; ++i) ws.dLatent[k][i] += ws.dFromPrediction[i];
```
*src/network.cpp:250 — `MuZeroNetwork::trainStep` (reverse pass, prediction head)*

`dLatent[k]` accumulates from two sources: the prediction head at step `k`,
and the dynamics step `k → k+1`. Going strictly downward guarantees the
second has already arrived before `dLatent[k]` is consumed — which is why
this is one reverse loop rather than two passes.

### Scaling rule one: the loss weights steps at 1/K

```cpp
        // Every unroll step past 0 contributes at 1/K, so a deeply
        // unrolled sample does not outweigh a shallow one.
        const float tailScale = (K > 0) ? 1.0f / static_cast<float>(K) : 1.0f;
        auto lossScale = [&](int k) { return k == 0 ? 1.0f : tailScale; };
```
*src/network.cpp:181 — `MuZeroNetwork::trainStep`*

Step 0 counts fully; steps 1 through `K` each count `1/K`. Without it, a
`K = 5` sample would contribute six times the gradient of a `K = 0` sample
purely because it is longer, and the batch's effective learning rate would
depend on the unroll depth.

### Scaling rule two: the half gradient

```cpp
                reluBackwardInto(ws.dynamicsPre[k - 1], ws.dDynamicsHidden, ws.dDynamicsPre);
                gFc1_.backwardInto(ws.dynamicsInput[k - 1], ws.dDynamicsPre, ws.dDynamicsInput);

                // The half gradient. Scaling what flows back into the
                // dynamics input keeps gradient magnitude from compounding
                // across the recurrence. See setDynamicsGradientScale.
                for (int i = 0; i < kLatentSize; ++i) {
                    ws.dLatent[k - 1][i] += dynamicsGradientScale_ * ws.dDynamicsInput[i];
                }
```
*src/network.cpp:282 — `MuZeroNetwork::trainStep` (the half gradient)*

Gradient flowing back through the recurrence is halved at every step. By
the time signal from step 5 reaches step 0 it has been multiplied by
`0.5⁵ = 1/32`. This is the recurrent-network analogue of the forward-side
`minMaxNormalize` from section 02: one keeps activations from compounding
forward, the other keeps gradients from compounding backward.

**And it is not the gradient of the loss.** That is worth stopping on. The
true derivative of this loss with respect to `hFc1_`'s weights does not
have a 0.5 in it anywhere. MuZero applies one anyway, deliberately, because
the true gradient is numerically worse behaved than the deliberately wrong
one. The header says so in as many words (`include/mz/network.hpp:79`):
a "DELIBERATE deviation from the true gradient," with a seam
(`setDynamicsGradientScale`) that exists for exactly one purpose.

### Why a wrong gradient here is the worst kind of bug

It does not crash. It does not produce a NaN. It does not even make the
loss go up. A gradient that is uniformly 0.79× too small is a gradient that
still points downhill — training proceeds, the loss falls, everything looks
fine, and the network is simply worse than it should be, forever.

So how do you catch it? The obvious test is: take an SGD step, and check
that the loss dropped by about what the gradient predicted. It does not
work, and the reason is a genuinely instructive trap, written out in the
test itself:

```cpp
    // The obvious alternative -- take an SGD step and confirm the loss
    // drops by roughly what the gradient predicts -- cannot work here.
    // To first order a step of size r drops the loss by r*|g|^2 -
    // (r^2/2)*g'Hg, so scaling the whole gradient by any constant scales
    // every measured drop by the same constant squared and cancels out of
    // any ratio of drops. A missing half gradient is exactly such a
    // rescaling, and would sail through.
```
*tests/test_network.cpp:299 — `test_gradient_matches_numerical_across_every_layer`*

Spelled out: if the true gradient is `g` and the code computes `c·g` for
some constant `c`, then the loss drop at step size `r` becomes
`rc|g|² − (r²c²/2)g'Hg`. Compare the drops at two step sizes and `c`
appears in both, in the same places, and cancels out of the ratio. The
check is *algebraically blind* to any uniform rescaling of the gradient —
which is precisely what a missing `1/K` or a missing half gradient is. The
test would pass. The network would train worse forever.

The check that does work perturbs **one parameter at a time** and compares
the resulting central difference against the gradient `trainStep`
accumulated for that exact slot:

```cpp
    const int unrollSteps = 3;
    std::vector<UnrolledSample> batch = {makeProbeSample(unrollSteps)};

    MuZeroNetwork network(8675309);
    network.setDynamicsGradientScale(1.0f);   // measure the TRUE gradient
    network.trainStep(batch, 0.0f);           // populates gradients, moves nothing
```
*tests/test_network.cpp:319 — `test_gradient_matches_numerical_across_every_layer`*

A per-slot comparison has no ratio for `c` to cancel out of: it is an
absolute claim about one number. The test sweeps a strided grid of slots
across **all eight layers**, requires at least one real comparison per layer
(a layer the reverse pass never reaches would have all-zero gradients and
would otherwise pass in silence), and skips slots below a noise floor
rather than comparing them against float noise.

**The half gradient has to be switched off for it**, and this surprises
people. A finite difference of the loss necessarily measures the *true*
gradient of the loss. The reverse pass, by design, does not compute the
true gradient. So with `dynamicsGradientScale_` left at 0.5 the two
disagree **by design** — measurably: the test's own comment records that
`gFc1` and `gFc2` come out at roughly 0.79–0.81 of the true gradient. That
is not a bug to be fixed; it is the deviation working. Hence the seam:
training always runs at 0.5, and the gradient check is the only caller that
ever sets it to 1.

Which leaves the 0.5 itself unguarded — so a third test pins it, by an
argument rather than by a tolerance. `test_half_gradient_scales_the_dynamics_input`
observes that the accumulated gradient is polynomial in the scale `s`, and
is *affine* in `s` for the dynamics layers exactly when `K = 2` and for the
representation layers exactly when `K = 1`. Where a function is affine,
`g(0.5)` must be the exact midpoint of `g(0)` and `g(1)` — an equality, and
one that fails for any other scale. The test measures all three and checks
the midpoint identity.

And the `1/K`? **No gradient check can see it at all.** `1/K` is part of the
loss *definition*, so dropping it rescales the loss and its gradient
together and every self-consistent check still agrees with itself. The only
way to catch it is to recompute the loss independently — which
`test_loss_uses_one_over_k_scaling` does, by walking the same forward pass
through the public `initialInference` and `recurrentInference` and never
going near `trainStep`'s arithmetic.

| what could be wrong | would `trainStep` still run? | caught by |
|---|---|---|
| sign error, unreached head | yes | `test_train_step_moves_downhill` (direction only) |
| any uniform rescale of the gradient | yes | per-parameter finite difference, `test_gradient_matches_numerical_across_every_layer` |
| half gradient present but not 0.5 | yes | the midpoint identity, `test_half_gradient_scales_the_dynamics_input` |
| missing `1/K` | yes | independent loss recomputation, `test_loss_uses_one_over_k_scaling` |

Four tests, four distinct blind spots, none of which subsumes another.

> **vs AlphaZero.** The sibling project's backward pass is one page: two
> heads into one hidden layer, one gradient per weight, one call per
> example. No recurrence, no unroll, no accumulation across time, and
> therefore neither scaling rule and none of the traps above. The moment a
> model is learned and rolled forward inside the loss, backprop-through-time
> and everything in this section arrives with it.

---

## 09 — The full loop

**Concept: the training loop, and reading an evaluation honestly**

```
   ┌──────────────┐      ┌─────────────────┐      ┌──────────────────┐
   │  self-play   │ ───▶ │  replay buffer  │ ───▶ │      train       │
   │ 25 games/it. │      │  2,000 GAMES    │      │ 40 × batch 64,   │
   │ 100 sims/move│      │ (not positions) │      │ K = 5 unroll     │
   └──────────────┘      └─────────────────┘      └────────┬─────────┘
          ▲                                                │
          └──────────── updated network plays next ────────┘
                                                            │
                                                  (every 10 iterations)
                                                            ▼
                                                    ┌────────────────┐
                                                    │    evaluate    │
                                                    │  vs. minimax   │
                                                    └────────────────┘
```

```cpp
    for (int iteration = 0; iteration < numIterations; ++iteration) {
        for (int g = 0; g < gamesPerIteration; ++g) {
            buffer.add(mz::playSelfPlayGame(network, selfPlayConfig, rng));
        }

        if (buffer.totalPositions() >= static_cast<std::size_t>(batchSize)) {
            mz::MuZeroNetwork::Losses losses;
            for (int step = 0; step < trainStepsPerIteration; ++step) {
                std::vector<mz::UnrolledSample> batch;
                batch.reserve(batchSize);
                for (const auto& sample : buffer.samplePositions(batchSize)) {
                    batch.push_back(mz::makeUnrolledSample(buffer.game(sample.gameIndex),
                                                           sample.position, targetConfig));
                }
                losses = network.trainStep(batch, learningRate);
            }
```
*apps/train.cpp:38 — the training loop*

The cycle: self-play produces trajectories; `makeUnrolledSample` turns a
sampled position of a trajectory into an observation, five actions and
aligned targets; `trainStep` turns a batch of those into gradients; the
better network produces better trajectories.

The replay buffer stores **whole games**, not flattened positions, and that
is forced rather than stylistic: both of MuZero's targets read *forward*
from a position. The K-step unroll needs the actions that followed; the
n-step value target needs rewards and search values from later in the same
game. A buffer of independent positions cannot answer either question.

The evaluate branch is the only part of this that is not the network
grading its own homework. `evaluateAgainstMinimax` plays the network
against an exhaustive solver, as both X and O, with greedy noise-free
search and a fixed RNG seed. A perfect player never loses, so **wins is
zero by construction** — across all 23 training runs and 150-odd
evaluations in this project, `wins` was `0` every single time. The only
number that can move is how often the network draws instead of losing.

### What actually happened

`docs/results.md` is the full record. The headline, stated plainly:

**The spec's first success criterion — converging to drawing play against
minimax — was not met.** Across 23 training runs, exactly one converged:

```
$ ./build/evaluate ckpt_sim100_it800_seed303.bin 50
vs minimax over 50 games/side: wins=0 draws=100 losses=0
```

That run was `numSimulations=100`, `temperatureMoves=6`, 800 iterations,
seed 303. The **other two seeds at that identical configuration** finished
at `draws=50 losses=50` and `draws=0 losses=100`. No configuration tried
converged on all three of its seeds. Tripling the search budget to 300
simulations, on the same three seeds with everything else held fixed,
took the cell from 1/3 converged to **0/3** — more search made it worse.

The sibling AlphaZero project reached `draws=40 losses=0`, every game, both
sides, after **20** iterations.

Here is the tail of the converged run's own training log — the same
`evalIntervalIterations = 10` eval that `apps/train.cpp` prints:

```
iteration 769: games=2000 positions=14903 loss=3.7282 (value=0.8379 policy=2.8235 reward=0.0668)
eval vs minimax: wins=0 draws=0 losses=40
iteration 779: games=2000 positions=14975 loss=4.2019 (value=1.1313 policy=2.9913 reward=0.0793)
eval vs minimax: wins=0 draws=40 losses=0
iteration 789: games=2000 positions=15059 loss=3.9431 (value=0.8765 policy=2.9941 reward=0.0725)
eval vs minimax: wins=0 draws=0 losses=40
iteration 799: games=2000 positions=15133 loss=3.9274 (value=1.0947 policy=2.7743 reward=0.0584)
eval vs minimax: wins=0 draws=40 losses=0
```

Read that carefully, because it is the most honest thing in this document.
This is the **successful** run. It is oscillating between perfect and
hopeless every ten iterations, right up to the final one. Over the whole
800-iteration run its 80 training-time evaluations came out: 63 at
`losses=40`, 10 at `losses=0`, 7 split at `draws=20 losses=20`. The
checkpoint that scores `draws=100 losses=0` is simply the one the last
window happened to land on. `apps/train.cpp` has no early stopping and no
best-checkpoint tracking, so "final" means "last," not "best."

Do not read this as *MuZero just needs more iterations here*. It is not a
tuning footnote. Two seeds at the same configuration and the same 800
iterations did not converge; `temperatureMoves=2` at 1500 iterations did
not converge on any of three seeds, and seed 303 at that setting scored
`losses=100` at both 400 **and** 1500 iterations, unchanged despite 3.75×
the training. The training loss was still falling at the end of every run
and the categorical score did not track it. What is missing is not
iterations. It is reliability.

> **vs AlphaZero.** Each MuZero iteration here does four times the
> gradient-step throughput of an AlphaZero iteration (40×64 samples versus
> 20×32), so 800 iterations is worth roughly 80× AlphaZero's total training
> samples — and it converged on one seed in three, where AlphaZero's much
> smaller budget converged in 20 iterations. Iteration counts are not a
> unit-comparable currency between the two projects (different
> architectures, different self-play loops, different target construction);
> nor is "converged" measured the same way — AlphaZero's number is a single
> reported run, not a seed grid — so treat both as context rather than a
> multiplier. The comparison that *is* precise is in the next section.

### A performance pass, and how it was checked

None of the above is what a profiler of this training loop would have led
with. A run of 200 iterations put roughly a quarter of wall-clock time
inside the allocator — `malloc`/`free`, not arithmetic — and four changes
went after it: `Dense::forwardInto`/`backwardInto` and the activation
helpers write into caller-owned buffers instead of returning a fresh
`std::vector` each call; `trainStep` reuses one scratch struct instead of
allocating on the order of sixty vectors per sample; MCTS nodes moved from
individually-`unique_ptr`-owned children into an arena addressed by integer
index, reused across searches, with one `MCTS` object per game instead of
per move; and `Board` maintains its outcome incrementally — rechecking only
the (at most four) lines through the square just played — while minimax
iterates squares directly instead of calling `legalMoves()` at every node
of the game tree, the single biggest win of the four. The buffers in the
new `MuZeroNetwork::Workspace` belong to the *caller* (`MCTS` holds one,
`trainStep` holds another) rather than to the network itself, deliberately,
so that inference stays safe to call concurrently on a shared network —
which is what parallel self-play would need. 200 iterations went from
24.45s to about 15.7s, and the allocator's share of profiled samples went
from 23.9% to 0.8%.

None of that is a claim to take on faith, and it would be a strange place
for a project this insistent on provenance to start. Every one of the four
changes was checked the same way: run training from a fixed seed before the
change and after it, and compare the resulting checkpoints byte-for-byte.
Training here is fully reproducible from a seed — the same number drives
network initialization, self-play, and every sampling decision — so if a
change to *how* a quantity is stored altered *what* it computes by so much
as one bit, the checkpoints would stop matching. They did not: two seeds
(777 and 4242) produced identical checkpoints through the whole pipeline
after each of the four changes, which is exactly the property this document
leans on everywhere else — that a run is a fact about a seed, not a fact
about that particular execution.

Two seeds is a spot check, though, not a proof, and the one bug this pass
actually introduced was in exactly the part a spot check cannot reliably
see. `Board`'s incremental outcome tracking works from a hand-written table
of which lines pass through which square (`kLinesThrough` in
`src/board.cpp`); a wrong entry in it is invisible until some specific game
reaches the one position it misjudges, and one slipped in while the table
was being written. What caught it was not the checkpoint comparison but
`test_cached_outcome_matches_a_full_rescan_everywhere`, which does not spot
check: it walks all 549946 nodes of the full tic-tac-toe game tree and
asserts, at every single one, that the incrementally maintained outcome
agrees with a full eight-line rescan. Two forms of the same idea — trust
nothing that was not checked against an independent computation of the same
answer — at two different scales.

---

## 10 — What this bought, and what it cost

**Concept: when a learned model is worth it**

On tic-tac-toe, MuZero is strictly worse than AlphaZero, and it is worth
being blunt about every dimension:

- **Slower to converge**, by a wide margin, when it converges at all.
- **More code**: three networks instead of one, a reward head, a
  trajectory-shaped replay buffer, an n-step target module,
  backpropagation through time with two scaling rules, `MinMaxStats`, and
  two diagnostic tools with no counterpart in the sibling project.
- **Less reliable**: 1 converged run out of 23.
- **And it spends all of that** laboriously rediscovering, from data, rules
  that were sitting in `Board` the whole time — which squares are occupied,
  when three in a row ends the game, who won.

### Where the failure actually is

`docs/results.md` localizes it, and the localization matters because it
distinguishes "this implementation is broken" from "this configuration is
unreliable."

- **The value head is correct.** Probed on hand-built positions, it reports
  roughly +0.54 to +0.65 where the player to move has an immediate win, and
  −0.41 to −0.66 where the player to move is in trouble. Right signs,
  sensible magnitudes, both directions. A sign error in the target, the
  backup, or the gradient would destroy this first. It did not.
- **The policy head misses immediate tactics.** On "X to move, and square 2
  wins outright," its argmax lands on square 5 or 8, with only 0.09 to 0.32
  of its mass on the winning move.
- **Search cannot rescue it**, for the reason in section 03: depth 18–19 on
  a 9-ply game means most of the budget is spent past the point where any
  legal game could still be running.

The observed losses are not diffuse weakness. Because
`evaluateAgainstMinimax` is fully deterministic, a `draws=50 losses=50`
result means *the network plays one side perfectly and walks into the exact
same losing line as the other side, fifty times out of fifty* — one wrong
move in one position, repeated. Both transcripts in `docs/results.md` fail
at the identical *kind* of ply: an immediate three-in-a-row that must be
blocked, with the visit distribution spread across candidates instead of
collapsed onto the one correct block (4% on the blocking square in one
case; 47% on a non-blocking move versus 41% on the correct one in the
other). The converged checkpoint puts 74–80% of its visits on the correct
block in the same class of position, and draws.

**The single converged run is genuine evidence that the implementation is
correct end to end.** Nothing that reaches `wins=0 draws=100 losses=0`
against an exhaustive solver, on both sides, has a sign error in its backup
or its targets or its gradient. What this project failed to produce is a
configuration that does that *reliably*. That is a reliability result, not a
structural defect — and it is a more useful thing to have measured than a
triumph would have been, because it is a direct, quantified answer to "what
does it cost to take the rules away," on a game small enough that the
comparison is unambiguous.

### One more measurement worth sitting with

`tools/latent_probe`, run on the **converged** checkpoint over 2000
trajectories:

| unroll depth k | samples | mean \|value error\| | mean policy distance |
|---|---|---|---|
| 1 | 2000 | 0.3128 | 0.2422 |
| 2 | 1952 | 0.2316 | 0.4124 |
| 3 | 1840 | 0.3268 | 0.4499 |
| 4 | 1606 | 0.2988 | 0.4725 |
| 5 | 1296 | 0.3371 | 0.5042 |
| 6 | 847 | 0.3305 | 0.5501 |

By depth 6 the imagined policy disagrees with the real game's policy over
half the time (0.55 total variation), and the terminal-reward sense is off
by 0.77 on rewards drawn from `{−1, 0, +1}` — in the checkpoint that plays
perfectly. So model fidelity at depth is *not* what determined convergence.
What determined it was whether the root-level policy and value were right
often enough. A converged network's shallow, frequently-visited estimates
are good enough that deep wrong simulations get outvoted; an unconverged
network has no such shallow anchor, and the same deep noise has nothing to
outvote it.

### What it buys, that AlphaZero structurally cannot have

Two things, and neither of them is visible on a 3×3 board.

**Domains with no simulator.** Everything in sections 01–08 runs without
ever calling `applyMove`, `legalMoves` or `isTerminal` below the root.
Point the same machinery at Atari frames, or at any environment you can
only act in and observe, and it still runs. AlphaZero's search cannot be
pointed anywhere its rules are not already written down.

**Reanalyze.** Because MuZero's value target is bootstrapped from its own
search rather than stamped with a finished game's outcome (section 07),
stored trajectories can be re-searched with a stronger network and their
targets overwritten in place — old data appreciating in value as the agent
improves. AlphaZero's targets are facts about the past and cannot be
improved by anything. This project does not implement Reanalyze, but three
decisions were made to leave room for it without restructuring, listed in
the spec's **Reanalyze Readiness** section: whole trajectories in the replay
buffer (including the raw observation at every step, which is what a fresh
search needs), search policies and values stored as mutable arrays with an
in-place replacement accessor (`ReplayBuffer::replaceSearchTargets`), and
target construction as a pure function of a trajectory rather than a step
fused into self-play.

The full side-by-side accounting of every difference —
given vs. learned dynamics, latent states and the loss of
interpretability, root-only legality, the reward head, bootstrapped vs.
outcome targets, trajectory vs. position storage, the two PUCT formulas,
and Reanalyze — is in
[`muzero-vs-alphazero.md`](muzero-vs-alphazero.md), along with the
deliberate deviations from the paper (scalar heads instead of the 601-bin
categorical support, SGD instead of Adam, no Reanalyze, small MLPs instead
of residual towers).

---

## 11 — Concept map

A quick index back into the repository, and across to the sibling project,
for whichever fundamental you want to see again in situ.

| Concept | In this codebase (MuZero) | In `alphazero-tictactoe` |
|---|---|---|
| Markov decision process (state / action / reward) | `mz::Board`, `mz::Cell`, `mz::Outcome` — `include/mz/board.hpp` | same, `az::Board` |
| State canonicalization | Player-relative `encode()`, used only at the search root — `src/board.cpp` | Player-relative `encode()`, used at every node — `src/board.cpp:68` |
| Model-based RL with a **given** model | — (deliberately absent) | `Board::applyMove` inside search — `src/mcts.cpp:83` |
| Model-based RL with a **learned** model | `MuZeroNetwork::recurrentInference` — `src/network.cpp:102` | — (structurally impossible) |
| Latent state, no decoder | `kLatentSize = 32`, class comment — `include/mz/network.hpp:17` | — |
| Policy π(a\|s) and value V(s) approximation | The prediction network `f` — `src/network.cpp:58` | `Network::predict` — `src/network.cpp:23` |
| Action encoding | One-hot concatenation — `src/network.cpp:42` | — (an action is a board index) |
| Recurrence stabilization | `minMaxNormalizeInto` — `src/mlp.cpp:154` | — |
| Planning / lookahead | `MCTS::run` — `src/mcts.cpp:136` | `MCTS::simulate`, `MCTS::run` — `src/mcts.cpp:73, 92` |
| Root-only legality | `maskAndRenormalize` at the root only — `src/mcts.cpp:78`, `src/mcts.cpp:146` | Legality at every node |
| Exploration vs. exploitation (search-time) | `explorationTerm` + `MinMaxStats` — `src/mcts.cpp:32`, `src/mcts.cpp:19` | Fixed `c_puct` PUCT — `src/mcts.cpp:50` |
| Value normalization in search | `MinMaxStats` — `src/mcts.cpp:9` | — (Q is already in `[-1, 1]`) |
| Guaranteed exploration / Dirichlet root noise | `MCTS::mixDirichletNoise`, self-play only — `src/mcts.cpp:78`, `src/selfplay.cpp:11` | `MCTS::mixDirichletNoise` — `src/mcts.cpp:13` |
| Exploration vs. exploitation (trajectory-time) | `SelfPlayConfig::temperatureMoves` — `include/mz/selfplay.hpp:33` | `SelfPlayConfig::temperatureMoves` |
| Credit assignment with intermediate rewards | `backupPath` — `src/mcts.cpp:37` | Sign-flip only, no reward — `src/mcts.cpp:86` |
| Reward as distinct from value | The dynamics reward head — `src/network.cpp:117`, `src/selfplay.cpp:40` | — (`Board::outcome()`) |
| Monte Carlo return | `tdSteps = 32` collapsing the target to the outcome — `include/mz/targets.hpp:16` | The `z` label — `src/selfplay.cpp:31` |
| Bootstrapping / TD learning | `bootstrappedValue` — `src/targets.cpp:18` | — |
| Policy improvement operator | Root visit distribution as the policy target — `src/selfplay.cpp:26` | Same — `src/selfplay.cpp:26` |
| Backpropagation through time | `MuZeroNetwork::trainStep` — `src/network.cpp:194` | — (one position, one backward pass) |
| Gradient-scaling rules | `1/K` — `src/network.cpp:211`; half gradient — `src/network.cpp:312` | — |
| Gradient verification | Per-parameter finite difference across all eight layers — `tests/test_network.cpp:296` | Per-weight gradient check — `tests/test_network.cpp` |
| Trajectory storage | `GameHistory`, `ReplayBuffer` over whole games — `include/mz/game_history.hpp`, `include/mz/replay_buffer.hpp` | Flat `TrainingExample` positions |
| Reanalyze hook (unused) | `ReplayBuffer::replaceSearchTargets` — `include/mz/replay_buffer.hpp:44` | — (structurally impossible) |
| Bias-free evaluation | `evaluateAgainstMinimax` — `src/eval.cpp:39`, `src/minimax.cpp` | Identical in shape — `src/eval.cpp` |
| Per-side diagnostic | One full game as X and one as O, with visit distributions — `tools/diag_eval.cpp` | `tools/diag_eval.cpp` |
| Model-fidelity diagnostic | Rolls `g` forward and compares against a fresh `h` on the real board — `tools/latent_probe.cpp` | — (no model to be wrong) |

---

Written against the source in this repository — build and run it yourself
with:

```sh
mkdir build && cd build && cmake .. && cmake --build .
./train 800 checkpoint.bin 303      # the third argument is the seed
./evaluate checkpoint.bin 50
```

Seed 303 reproduces the one converged run exactly. Omit the seed and, on
the evidence in `docs/results.md`, you have roughly one-in-three odds.

Mark Castelluccio, designed and implemented with
[Claude Code](https://claude.com/claude-code).
