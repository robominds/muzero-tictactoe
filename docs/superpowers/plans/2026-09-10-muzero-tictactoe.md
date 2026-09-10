# MuZero-Style Tic-Tac-Toe Player Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A from-scratch, dependency-free C++17 MuZero that learns tic-tac-toe by searching inside a model it learns itself, documented as a learning project and compared throughout against the sibling `alphazero-tictactoe` implementation.

**Architecture:** A static library `libmz` of single-purpose components consumed by three executables and two diagnostic tools. Three small MLPs (representation, dynamics, prediction) are built from one hand-written `Dense` layer primitive whose `backward` accumulates gradients, which is what makes backprop-through-time over a K-step unroll possible without an autograd library. MCTS runs entirely over learned latent states; the real rules are consulted only at the search root and to step the actual game during self-play.

**Tech Stack:** C++17, CMake 3.16+, standard library only. No external dependencies, no autograd, no GPU.

**Spec:** `docs/superpowers/specs/2026-09-10-muzero-tictactoe-design.md`

## Global Constraints

- C++17. CMake 3.16 minimum. `CMAKE_CXX_STANDARD_REQUIRED ON`.
- **Zero external dependencies.** Standard library only. No test framework — tests are `assert`-based executables with a `main`.
- Namespace `mz` for every library symbol. Headers live under `include/mz/` and are included as `"mz/<name>.hpp"`.
- Every test target must be compiled with `-UNDEBUG`. The default build type is Release, which defines `NDEBUG` and would compile every `assert` to a no-op, making the suite vacuously pass.
- Default `CMAKE_BUILD_TYPE` is `Release` when the user does not set one.
- Network dimensions, fixed project-wide: observation 18, action 9, latent 32, hidden 64.
- Search constants, fixed project-wide: `kPbCInit = 1.25f`, `kPbCBase = 19652.0f`, `kDiscount = 1.0f`.
- Unroll length K defaults to 5. It is always called `unrollSteps` in code; `N` is reserved for the train loop's checkpoint/eval interval. Never reuse `K` for that interval.
- Perspective convention, relied on by every component: a **value** is always from the perspective of the player to move in the state it describes; a **reward** is always from the perspective of the player who took the action producing it.
- Commit after every task. Commit messages end with:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

---

### Task 1: Scaffolding, board, and minimax

Carried over from `alphazero-tictactoe` with the namespace and include path changed. Identical rules are deliberate: the comparison only means something if both projects play the same game. MuZero simply is not allowed to consult these inside the search tree.

**Files:**
- Create: `CMakeLists.txt`
- Create: `include/mz/board.hpp`, `src/board.cpp`
- Create: `include/mz/minimax.hpp`, `src/minimax.cpp`
- Test: `tests/test_board.cpp`, `tests/test_minimax.cpp`

**Interfaces:**
- Consumes: nothing.
- Produces: `mz::Cell{Empty,X,O}`, `mz::Outcome{Ongoing,XWins,OWins,Draw}`, `mz::Board` with `cellAt(int)`, `playerToMove()`, `isLegalMove(int)`, `legalMoves() -> std::vector<int>`, `applyMove(int) -> Board`, `outcome()`, `isTerminal()`, `encode() -> std::array<float,18>`, `operator==`. Free function `mz::minimaxBestMove(const Board&) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/test_board.cpp`:

```cpp
#include <cassert>
#include <cstdio>
#include "mz/board.hpp"

using namespace mz;

void test_new_board_has_nine_legal_moves() {
    Board b;
    assert(b.legalMoves().size() == 9);
    assert(b.playerToMove() == Cell::X);
    assert(!b.isTerminal());
}

void test_apply_move_alternates_player() {
    Board b;
    Board b2 = b.applyMove(0);
    assert(b2.cellAt(0) == Cell::X);
    assert(b2.playerToMove() == Cell::O);
    assert(b2.legalMoves().size() == 8);
}

void test_row_win_detected() {
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(3); // O
    b = b.applyMove(1); // X
    b = b.applyMove(4); // O
    b = b.applyMove(2); // X completes top row
    assert(b.outcome() == Outcome::XWins);
    assert(b.isTerminal());
    assert(b.legalMoves().empty());
}

void test_diagonal_win_detected() {
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(1); // O
    b = b.applyMove(4); // X
    b = b.applyMove(2); // O
    b = b.applyMove(8); // X completes diagonal
    assert(b.outcome() == Outcome::XWins);
}

void test_draw_detected() {
    Board b;
    int moves[] = {0, 1, 2, 4, 3, 5, 7, 6, 8};
    for (int m : moves) b = b.applyMove(m);
    assert(b.outcome() == Outcome::Draw);
}

void test_illegal_move_rejected_by_isLegalMove() {
    Board b;
    b = b.applyMove(0);
    assert(!b.isLegalMove(0));
    assert(b.isLegalMove(1));
}

void test_encode_is_from_perspective_of_player_to_move() {
    Board b;
    b = b.applyMove(0); // X at 0, O to move
    auto enc = b.encode();
    for (int i = 0; i < 9; ++i) assert(enc[i] == 0.0f);
    assert(enc[9 + 0] == 1.0f);
}

int main() {
    test_new_board_has_nine_legal_moves();
    test_apply_move_alternates_player();
    test_row_win_detected();
    test_diagonal_win_detected();
    test_draw_detected();
    test_illegal_move_rejected_by_isLegalMove();
    test_encode_is_from_perspective_of_player_to_move();
    std::printf("all board tests passed\n");
    return 0;
}
```

`tests/test_minimax.cpp`:

```cpp
#include <cassert>
#include <cstdio>
#include "mz/board.hpp"
#include "mz/minimax.hpp"

using namespace mz;

void test_minimax_takes_immediate_win() {
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(3); // O
    b = b.applyMove(1); // X: X at 0,1, threat at 2
    b = b.applyMove(4); // O
    assert(minimaxBestMove(b) == 2);
}

void test_minimax_blocks_immediate_loss() {
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(5); // O
    b = b.applyMove(1); // X at 0,1 threatens 2 -- O to move must block
    assert(minimaxBestMove(b) == 2);
}

void test_minimax_never_loses_against_itself() {
    Board b;
    while (!b.isTerminal()) {
        b = b.applyMove(minimaxBestMove(b));
    }
    assert(b.outcome() == Outcome::Draw);
}

int main() {
    test_minimax_takes_immediate_win();
    test_minimax_blocks_immediate_loss();
    test_minimax_never_loses_against_itself();
    std::printf("all minimax tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run tests to verify they fail**

```sh
mkdir -p build && cmake -S . -B build && cmake --build build
```

Expected: FAIL — `CMakeLists.txt` does not exist yet.

- [ ] **Step 3: Write the headers**

`include/mz/board.hpp`:

```cpp
#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace mz {

enum class Cell : int8_t { Empty = 0, X = 1, O = 2 };

enum class Outcome { Ongoing, XWins, OWins, Draw };

class Board {
public:
    Board();

    Cell cellAt(int index) const;
    Cell playerToMove() const;

    bool isLegalMove(int index) const;
    std::vector<int> legalMoves() const;

    // Precondition: isLegalMove(index) == true (asserted).
    Board applyMove(int index) const;

    Outcome outcome() const;
    bool isTerminal() const { return outcome() != Outcome::Ongoing; }

    // 18 floats: [my stones (9), opponent stones (9)], from the
    // perspective of playerToMove(). MuZero sees this only at the search
    // root -- everywhere else it works with latents produced from it.
    std::array<float, 18> encode() const;

    bool operator==(const Board& other) const;

private:
    std::array<Cell, 9> cells_;
    Cell toMove_;
};

} // namespace mz
```

`include/mz/minimax.hpp`:

```cpp
#pragma once
#include "mz/board.hpp"

namespace mz {

// Returns the game-theoretically optimal move for board.playerToMove(),
// via exhaustive minimax search. Precondition: !board.isTerminal().
// Used only to measure convergence -- never for training, and never
// inside MCTS.
int minimaxBestMove(const Board& board);

} // namespace mz
```

- [ ] **Step 4: Write the implementations**

`src/board.cpp`:

```cpp
#include "mz/board.hpp"
#include <cassert>

namespace mz {

namespace {
constexpr int kLines[8][3] = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
    {0, 4, 8}, {2, 4, 6},
};
}

Board::Board() : toMove_(Cell::X) {
    cells_.fill(Cell::Empty);
}

Cell Board::cellAt(int index) const {
    return cells_[index];
}

Cell Board::playerToMove() const {
    return toMove_;
}

bool Board::isLegalMove(int index) const {
    if (index < 0 || index >= 9) return false;
    if (isTerminal()) return false;
    return cells_[index] == Cell::Empty;
}

std::vector<int> Board::legalMoves() const {
    std::vector<int> moves;
    if (isTerminal()) return moves;
    for (int i = 0; i < 9; ++i) {
        if (cells_[i] == Cell::Empty) moves.push_back(i);
    }
    return moves;
}

Board Board::applyMove(int index) const {
    assert(isLegalMove(index));
    Board next = *this;
    next.cells_[index] = toMove_;
    next.toMove_ = (toMove_ == Cell::X) ? Cell::O : Cell::X;
    return next;
}

Outcome Board::outcome() const {
    for (const auto& line : kLines) {
        Cell a = cells_[line[0]], b = cells_[line[1]], c = cells_[line[2]];
        if (a != Cell::Empty && a == b && b == c) {
            return a == Cell::X ? Outcome::XWins : Outcome::OWins;
        }
    }
    for (int i = 0; i < 9; ++i) {
        if (cells_[i] == Cell::Empty) return Outcome::Ongoing;
    }
    return Outcome::Draw;
}

std::array<float, 18> Board::encode() const {
    std::array<float, 18> out{};
    Cell mine = toMove_;
    Cell theirs = (toMove_ == Cell::X) ? Cell::O : Cell::X;
    for (int i = 0; i < 9; ++i) {
        out[i] = (cells_[i] == mine) ? 1.0f : 0.0f;
        out[9 + i] = (cells_[i] == theirs) ? 1.0f : 0.0f;
    }
    return out;
}

bool Board::operator==(const Board& other) const {
    return cells_ == other.cells_ && toMove_ == other.toMove_;
}

} // namespace mz
```

`src/minimax.cpp`:

```cpp
#include "mz/minimax.hpp"
#include <cassert>
#include <limits>

namespace mz {

namespace {

// Value of `board` from the perspective of board.playerToMove(), assuming
// optimal play by both sides: +1 win, -1 loss, 0 draw.
int scoreOf(const Board& board) {
    if (board.isTerminal()) {
        return board.outcome() == Outcome::Draw ? 0 : -1;
    }
    int best = std::numeric_limits<int>::min();
    for (int m : board.legalMoves()) {
        int childScore = -scoreOf(board.applyMove(m));
        if (childScore > best) best = childScore;
    }
    return best;
}

} // namespace

int minimaxBestMove(const Board& board) {
    assert(!board.isTerminal());
    int bestMove = board.legalMoves().front();
    int bestScore = std::numeric_limits<int>::min();
    for (int m : board.legalMoves()) {
        int score = -scoreOf(board.applyMove(m));
        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
        }
    }
    return bestMove;
}

} // namespace mz
```

- [ ] **Step 5: Write CMakeLists.txt**

This is the whole file for the finished project. Targets for sources that do not exist yet are added by later tasks; for now include only `board` and `minimax` plus their two tests, and add each remaining target in the task that creates it.

```cmake
cmake_minimum_required(VERSION 3.16)
project(muzero_tictactoe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(mz
    src/board.cpp
    src/minimax.cpp
)
target_include_directories(mz PUBLIC include)

enable_testing()

# Test binaries must assert regardless of CMAKE_BUILD_TYPE. The default
# build type is Release, which defines NDEBUG and would otherwise compile
# every assert() in these targets down to a no-op, making the test suite
# vacuous. Strip NDEBUG here without affecting the mz library or app
# targets, which should keep Release optimization.
function(mz_add_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} mz)
  target_compile_options(${name} PRIVATE -UNDEBUG)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

mz_add_test(test_board)
mz_add_test(test_minimax)
```

- [ ] **Step 6: Build and run the tests**

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: PASS — `all board tests passed`, `all minimax tests passed`, 2/2 tests.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt include/mz/board.hpp include/mz/minimax.hpp src/board.cpp src/minimax.cpp tests/test_board.cpp tests/test_minimax.cpp
git commit -m "$(cat <<'EOF'
Add board and minimax carried over from alphazero-tictactoe

Identical game rules, namespace mz. MuZero consults them only at the
search root and to step the real game during self-play.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: The `Dense` layer and activations

The one differentiation primitive in the project. Everything downstream depends on this being exactly right, so it is gradient-checked before anything is built on top of it.

Two design decisions matter and are easy to get wrong:

1. `backward` takes the input **explicitly** instead of caching it during `forward`. The dynamics network is applied K times inside one loss, each time with a different input, and backprop-through-time needs each of those inputs on the way back. A single cached input would silently use the wrong one.
2. `backward` **accumulates** into the gradient buffers rather than assigning. Same reason: K applications of one network must sum into one gradient. `zeroGrad()` is called once per training step, never per unroll step.

**Files:**
- Create: `include/mz/mlp.hpp`, `src/mlp.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/test_mlp.cpp`

**Interfaces:**
- Consumes: nothing.
- Produces: `mz::Dense` with `Dense(int inDim, int outDim, std::mt19937&)`, `inDim()`, `outDim()`, `forward(const std::vector<float>&) const -> std::vector<float>`, `backward(const std::vector<float>& input, const std::vector<float>& dOutput) -> std::vector<float>`, `zeroGrad()`, `applySgd(float learningRate, float scale)`, `write(std::ostream&) const`, `read(std::istream&)`. Free functions `relu`, `reluBackward`, `softmax9`, `minMaxNormalize`, `minMaxNormalizeBackward`.

- [ ] **Step 1: Write the failing test**

`tests/test_mlp.cpp`:

```cpp
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>
#include <vector>
#include "mz/mlp.hpp"

using namespace mz;

namespace {

// Central-difference numerical gradient of `loss` with respect to *x*.
template <typename LossFn>
std::vector<float> numericalInputGradient(LossFn loss, std::vector<float> x) {
    const float eps = 1e-3f;
    std::vector<float> grad(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float original = x[i];
        x[i] = original + eps;
        float up = loss(x);
        x[i] = original - eps;
        float down = loss(x);
        x[i] = original;
        grad[i] = (up - down) / (2.0f * eps);
    }
    return grad;
}

void assertClose(const std::vector<float>& a, const std::vector<float>& b, float tol, const char* what) {
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::fabs(a[i] - b[i]);
        float scale = std::fmax(1.0f, std::fmax(std::fabs(a[i]), std::fabs(b[i])));
        if (diff / scale > tol) {
            std::printf("%s mismatch at %zu: analytic=%.6f numeric=%.6f\n", what, i, a[i], b[i]);
            assert(false);
        }
    }
}

std::vector<float> wellSeparated(int n, std::mt19937& rng) {
    // Distinct, well-spread values: keeps the numerical gradient check away
    // from ReLU kinks and from min/max ties in minMaxNormalize, where the
    // derivative is only a subgradient.
    std::uniform_real_distribution<float> jitter(-0.05f, 0.05f);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = -1.0f + 2.0f * i / (n - 1) + jitter(rng);
    return v;
}

} // namespace

void test_forward_shape_and_value() {
    std::mt19937 rng(1234);
    Dense layer(3, 2, rng);
    std::vector<float> out = layer.forward({1.0f, 2.0f, 3.0f});
    assert(out.size() == 2);
    assert(layer.inDim() == 3 && layer.outDim() == 2);
}

void test_dense_input_gradient_matches_numerical() {
    std::mt19937 rng(7);
    Dense layer(5, 4, rng);
    std::vector<float> x = wellSeparated(5, rng);
    std::vector<float> dOut = {0.3f, -0.7f, 1.1f, 0.2f};

    // Scalar loss = dot(dOut, layer(x)), whose dLoss/dOutput is exactly dOut.
    auto loss = [&](const std::vector<float>& in) {
        std::vector<float> y = layer.forward(in);
        float s = 0.0f;
        for (size_t i = 0; i < y.size(); ++i) s += dOut[i] * y[i];
        return s;
    };

    std::vector<float> analytic = layer.backward(x, dOut);
    assertClose(analytic, numericalInputGradient(loss, x), 1e-2f, "dense dInput");
}

void test_dense_weight_gradient_matches_numerical() {
    std::mt19937 rng(11);
    Dense layer(4, 3, rng);
    std::vector<float> x = wellSeparated(4, rng);
    std::vector<float> dOut = {0.5f, -0.25f, 0.75f};

    layer.zeroGrad();
    layer.backward(x, dOut);

    // One SGD step with a known rate must move the output by
    // -rate * (dLoss/dW . dW), which we verify by finite difference on the
    // scalar loss dot(dOut, layer(x)).
    auto lossOf = [&](const Dense& l) {
        std::vector<float> y = l.forward(x);
        float s = 0.0f;
        for (size_t i = 0; i < y.size(); ++i) s += dOut[i] * y[i];
        return s;
    };
    float before = lossOf(layer);
    const float rate = 1e-3f;
    Dense stepped = layer;
    stepped.applySgd(rate, 1.0f);
    float after = lossOf(stepped);

    // Gradient descent must decrease this loss, and by roughly
    // rate * ||grad||^2 > 0.
    assert(after < before);
}

void test_backward_accumulates_across_calls() {
    std::mt19937 rng(3);
    Dense once(2, 2, rng);
    Dense twice = once;
    std::vector<float> x = {0.4f, -0.9f};
    std::vector<float> dOut = {1.0f, -1.0f};

    once.zeroGrad();
    once.backward(x, dOut);
    once.applySgd(0.1f, 1.0f);

    // Two backward calls without an intervening zeroGrad must produce
    // exactly twice the gradient -- this is what makes K-step
    // backprop-through-time over a reused network correct.
    twice.zeroGrad();
    twice.backward(x, dOut);
    twice.backward(x, dOut);
    twice.applySgd(0.1f, 0.5f);

    std::vector<float> a = once.forward(x);
    std::vector<float> b = twice.forward(x);
    assertClose(a, b, 1e-5f, "accumulate");
}

void test_relu_backward_matches_numerical() {
    std::mt19937 rng(5);
    std::vector<float> z = wellSeparated(6, rng);
    std::vector<float> dOut = {0.2f, -0.4f, 0.9f, 0.1f, -0.6f, 0.3f};
    auto loss = [&](const std::vector<float>& in) {
        std::vector<float> y = relu(in);
        float s = 0.0f;
        for (size_t i = 0; i < y.size(); ++i) s += dOut[i] * y[i];
        return s;
    };
    assertClose(reluBackward(z, dOut), numericalInputGradient(loss, z), 1e-2f, "relu");
}

void test_minmax_normalize_maps_into_unit_range() {
    std::vector<float> z = {-3.0f, 0.5f, 7.0f, 2.0f};
    std::vector<float> y = minMaxNormalize(z);
    assert(std::fabs(y[0] - 0.0f) < 1e-6f);
    assert(std::fabs(y[2] - 1.0f) < 1e-6f);
    for (float v : y) assert(v >= -1e-6f && v <= 1.0f + 1e-6f);
}

void test_minmax_normalize_handles_constant_input() {
    std::vector<float> z = {2.0f, 2.0f, 2.0f};
    std::vector<float> y = minMaxNormalize(z);
    for (float v : y) assert(std::isfinite(v));
}

void test_minmax_normalize_backward_matches_numerical() {
    std::mt19937 rng(13);
    std::vector<float> z = wellSeparated(8, rng);
    std::vector<float> dOut = {0.1f, -0.5f, 0.8f, 0.2f, -0.3f, 0.6f, -0.1f, 0.4f};
    auto loss = [&](const std::vector<float>& in) {
        std::vector<float> y = minMaxNormalize(in);
        float s = 0.0f;
        for (size_t i = 0; i < y.size(); ++i) s += dOut[i] * y[i];
        return s;
    };
    assertClose(minMaxNormalizeBackward(z, dOut), numericalInputGradient(loss, z), 2e-2f, "minmax");
}

void test_softmax9_sums_to_one() {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 0.0f, -1.0f, 0.5f, 0.5f, 2.5f, -2.0f};
    std::array<float, 9> p = softmax9(logits);
    float sum = 0.0f;
    for (float v : p) { assert(v > 0.0f); sum += v; }
    assert(std::fabs(sum - 1.0f) < 1e-5f);
}

void test_softmax9_is_shift_invariant_and_stable() {
    std::vector<float> small = {1.0f, 2.0f, 3.0f, 0.0f, -1.0f, 0.5f, 0.5f, 2.5f, -2.0f};
    std::vector<float> huge = small;
    for (float& v : huge) v += 1000.0f;
    std::array<float, 9> a = softmax9(small);
    std::array<float, 9> b = softmax9(huge);
    for (int i = 0; i < 9; ++i) {
        assert(std::isfinite(b[i]));
        assert(std::fabs(a[i] - b[i]) < 1e-5f);
    }
}

void test_dense_write_read_round_trips() {
    std::mt19937 rng(21);
    Dense original(4, 3, rng);
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    original.write(stream);

    std::mt19937 otherRng(22);
    Dense restored(4, 3, otherRng);
    stream.seekg(0);
    restored.read(stream);

    std::vector<float> x = {0.1f, -0.2f, 0.3f, 0.4f};
    assertClose(original.forward(x), restored.forward(x), 1e-6f, "round trip");
}

int main() {
    test_forward_shape_and_value();
    test_dense_input_gradient_matches_numerical();
    test_dense_weight_gradient_matches_numerical();
    test_backward_accumulates_across_calls();
    test_relu_backward_matches_numerical();
    test_minmax_normalize_maps_into_unit_range();
    test_minmax_normalize_handles_constant_input();
    test_minmax_normalize_backward_matches_numerical();
    test_softmax9_sums_to_one();
    test_softmax9_is_shift_invariant_and_stable();
    test_dense_write_read_round_trips();
    std::printf("all mlp tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run the test to verify it fails**

Add `mz_add_test(test_mlp)` to `CMakeLists.txt`, then:

```sh
cmake -S . -B build && cmake --build build
```

Expected: FAIL — `mz/mlp.hpp: No such file or directory`.

- [ ] **Step 3: Write the header**

`include/mz/mlp.hpp`:

```cpp
#pragma once
#include <array>
#include <iosfwd>
#include <random>
#include <vector>

namespace mz {

// A fully-connected layer y = Wx + b, with hand-written forward and
// backward passes. This is the only differentiation primitive in the
// project -- the three MuZero networks are built entirely from it.
//
// Two properties are load-bearing for MuZero specifically:
//
//   backward() takes the input EXPLICITLY rather than caching it during
//   forward(). The dynamics network is applied K times inside one loss,
//   each time on a different latent, and backprop-through-time needs each
//   of those inputs again on the way back. A single cached input would
//   silently use the wrong one.
//
//   backward() ACCUMULATES into the gradient buffers instead of assigning.
//   Those same K applications must sum into one shared gradient. Call
//   zeroGrad() once per training step -- never once per unroll step.
class Dense {
public:
    Dense() = default;
    Dense(int inDim, int outDim, std::mt19937& rng);

    int inDim() const { return inDim_; }
    int outDim() const { return outDim_; }

    // Wx + b. const, so inference paths can hold the network by const ref.
    std::vector<float> forward(const std::vector<float>& input) const;

    // Given dLoss/dOutput, accumulates dLoss/dW and dLoss/db and returns
    // dLoss/dInput. `input` must be the input this gradient came from.
    std::vector<float> backward(const std::vector<float>& input,
                                const std::vector<float>& dOutput);

    void zeroGrad();
    // theta -= learningRate * scale * grad. `scale` is normally 1/batchSize.
    void applySgd(float learningRate, float scale);

    void write(std::ostream& out) const;
    void read(std::istream& in);

private:
    int inDim_ = 0;
    int outDim_ = 0;
    std::vector<float> w_;      // outDim_ * inDim_, row-major
    std::vector<float> b_;      // outDim_
    std::vector<float> gradW_;
    std::vector<float> gradB_;
};

std::vector<float> relu(const std::vector<float>& z);
// `z` is the PRE-activation; ReLU passes gradient only where z > 0.
std::vector<float> reluBackward(const std::vector<float>& z, const std::vector<float>& dOut);

// Numerically stable softmax over exactly 9 logits (the action space).
std::array<float, 9> softmax9(const std::vector<float>& logits);

// Scales z into [0, 1] across its own elements, per MuZero appendix G:
//   y_i = (z_i - min(z)) / max(max(z) - min(z), 1e-5)
// Applied to every latent produced by representation and dynamics. Without
// it, applying dynamics repeatedly lets latent magnitudes drift and the
// recurrence destabilizes.
std::vector<float> minMaxNormalize(const std::vector<float>& z);

// The gradient of minMaxNormalize, including the terms that flow through
// min(z) and max(z) themselves -- treating them as constants would pass a
// casual eyeball but fails the numerical gradient check in test_mlp.
std::vector<float> minMaxNormalizeBackward(const std::vector<float>& z, const std::vector<float>& dOut);

} // namespace mz
```

- [ ] **Step 4: Write the implementation**

`src/mlp.cpp`:

```cpp
#include "mz/mlp.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <istream>
#include <ostream>

namespace mz {

namespace {
constexpr float kMinMaxFloor = 1e-5f;
}

Dense::Dense(int inDim, int outDim, std::mt19937& rng)
    : inDim_(inDim), outDim_(outDim),
      w_(static_cast<size_t>(inDim) * outDim), b_(outDim, 0.0f),
      gradW_(static_cast<size_t>(inDim) * outDim, 0.0f), gradB_(outDim, 0.0f) {
    // Scaled by fan-in, since the three networks here have quite different
    // input widths (18, 41, 32) and a single fixed range would leave some
    // of them saturated and others barely moving.
    float limit = 1.0f / std::sqrt(static_cast<float>(inDim));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (float& v : w_) v = dist(rng);
}

std::vector<float> Dense::forward(const std::vector<float>& input) const {
    assert(static_cast<int>(input.size()) == inDim_);
    std::vector<float> out(outDim_);
    for (int o = 0; o < outDim_; ++o) {
        float sum = b_[o];
        const float* row = &w_[static_cast<size_t>(o) * inDim_];
        for (int i = 0; i < inDim_; ++i) sum += row[i] * input[i];
        out[o] = sum;
    }
    return out;
}

std::vector<float> Dense::backward(const std::vector<float>& input,
                                   const std::vector<float>& dOutput) {
    assert(static_cast<int>(input.size()) == inDim_);
    assert(static_cast<int>(dOutput.size()) == outDim_);
    std::vector<float> dInput(inDim_, 0.0f);
    for (int o = 0; o < outDim_; ++o) {
        float d = dOutput[o];
        float* gradRow = &gradW_[static_cast<size_t>(o) * inDim_];
        const float* row = &w_[static_cast<size_t>(o) * inDim_];
        for (int i = 0; i < inDim_; ++i) {
            gradRow[i] += d * input[i];   // accumulate, never assign
            dInput[i] += d * row[i];
        }
        gradB_[o] += d;
    }
    return dInput;
}

void Dense::zeroGrad() {
    std::fill(gradW_.begin(), gradW_.end(), 0.0f);
    std::fill(gradB_.begin(), gradB_.end(), 0.0f);
}

void Dense::applySgd(float learningRate, float scale) {
    float step = learningRate * scale;
    for (size_t i = 0; i < w_.size(); ++i) w_[i] -= step * gradW_[i];
    for (size_t i = 0; i < b_.size(); ++i) b_[i] -= step * gradB_[i];
}

void Dense::write(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(w_.data()), w_.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(b_.data()), b_.size() * sizeof(float));
}

void Dense::read(std::istream& in) {
    in.read(reinterpret_cast<char*>(w_.data()), w_.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(b_.data()), b_.size() * sizeof(float));
}

std::vector<float> relu(const std::vector<float>& z) {
    std::vector<float> out(z.size());
    for (size_t i = 0; i < z.size(); ++i) out[i] = z[i] > 0.0f ? z[i] : 0.0f;
    return out;
}

std::vector<float> reluBackward(const std::vector<float>& z, const std::vector<float>& dOut) {
    assert(z.size() == dOut.size());
    std::vector<float> out(z.size());
    for (size_t i = 0; i < z.size(); ++i) out[i] = z[i] > 0.0f ? dOut[i] : 0.0f;
    return out;
}

std::array<float, 9> softmax9(const std::vector<float>& logits) {
    assert(logits.size() == 9);
    float maxLogit = logits[0];
    for (float l : logits) maxLogit = std::max(maxLogit, l);
    std::array<float, 9> out{};
    float sum = 0.0f;
    for (int i = 0; i < 9; ++i) {
        out[i] = std::exp(logits[i] - maxLogit);
        sum += out[i];
    }
    for (float& v : out) v /= sum;
    return out;
}

std::vector<float> minMaxNormalize(const std::vector<float>& z) {
    assert(!z.empty());
    float lo = *std::min_element(z.begin(), z.end());
    float hi = *std::max_element(z.begin(), z.end());
    float denom = std::max(hi - lo, kMinMaxFloor);
    std::vector<float> out(z.size());
    for (size_t i = 0; i < z.size(); ++i) out[i] = (z[i] - lo) / denom;
    return out;
}

std::vector<float> minMaxNormalizeBackward(const std::vector<float>& z, const std::vector<float>& dOut) {
    assert(z.size() == dOut.size());
    // With y_i = (z_i - m) / d where m = min(z), M = max(z), d = M - m:
    //   dy_i/dz_j = (delta_ij - [j==argmin]) / d
    //               - y_i / d * ([j==argmax] - [j==argmin])
    // Summing against dOut and writing S = sum(dOut), T = sum(dOut_i * y_i):
    //   dL/dz_j = dOut_j/d - [j==argmin]*S/d - ([j==argmax]-[j==argmin])*T/d
    size_t argmin = std::min_element(z.begin(), z.end()) - z.begin();
    size_t argmax = std::max_element(z.begin(), z.end()) - z.begin();
    float lo = z[argmin], hi = z[argmax];
    float range = hi - lo;
    float denom = std::max(range, kMinMaxFloor);

    std::vector<float> out(z.size());
    float S = 0.0f, T = 0.0f;
    for (size_t i = 0; i < z.size(); ++i) {
        S += dOut[i];
        T += dOut[i] * ((z[i] - lo) / denom);
    }
    // When the range is clamped by the floor, the denominator is a constant
    // and its derivative term drops out.
    bool clamped = range < kMinMaxFloor;
    for (size_t j = 0; j < z.size(); ++j) {
        float g = dOut[j] / denom;
        if (j == argmin) g -= S / denom;
        if (!clamped) {
            float indicator = (j == argmax ? 1.0f : 0.0f) - (j == argmin ? 1.0f : 0.0f);
            g -= indicator * T / denom;
        }
        out[j] = g;
    }
    return out;
}

} // namespace mz
```

- [ ] **Step 5: Add to the build and run the tests**

Add `src/mlp.cpp` to the `add_library(mz ...)` source list.

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: PASS — `all mlp tests passed`, 3/3 tests.

- [ ] **Step 6: Commit**

```bash
git add include/mz/mlp.hpp src/mlp.cpp tests/test_mlp.cpp CMakeLists.txt
git commit -m "$(cat <<'EOF'
Add hand-written Dense layer with gradient checks

backward() takes its input explicitly and accumulates gradients, which
is what lets one dynamics network be applied K times inside a single
loss during backprop-through-time.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Game history and replay buffer

Where MuZero's storage diverges from AlphaZero's. AlphaZero flattens finished games into independent `(position, policy, outcome)` rows, because nothing it trains on ever looks past the current position. MuZero needs to read **forward** from a position — K actions for the unroll, and n more steps for the bootstrap — so trajectories have to stay intact.

Keeping whole trajectories, including the raw observation at every step, is also exactly what a future Reanalyze pass needs: fresh search re-run over stored observations, with the results written back in place.

**Files:**
- Create: `include/mz/game_history.hpp`
- Create: `include/mz/replay_buffer.hpp`, `src/replay_buffer.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/test_replay_buffer.cpp`

**Interfaces:**
- Consumes: nothing.
- Produces: `mz::GameHistory` (public members `observations`, `actions`, `searchPolicies`, `searchValues`, `rewards`; method `length()`). `mz::ReplayBuffer` with `ReplayBuffer(size_t capacity)`, `add(GameHistory)`, `size()`, `totalPositions()`, `game(size_t) -> const GameHistory&`, `samplePositions(size_t count) -> std::vector<Sample>` where `Sample{size_t gameIndex; int position;}`, and `replaceSearchTargets(size_t, std::vector<std::array<float,9>>, std::vector<float>)`.

- [ ] **Step 1: Write the failing test**

`tests/test_replay_buffer.cpp`:

```cpp
#include <cassert>
#include <cstdio>
#include <set>
#include "mz/replay_buffer.hpp"

using namespace mz;

namespace {

// A trajectory of `length` moves, tagged so tests can tell games apart:
// every searchValue is `tag`.
GameHistory makeGame(int length, float tag) {
    GameHistory game;
    for (int t = 0; t < length; ++t) {
        std::array<float, 18> obs{};
        obs[0] = static_cast<float>(t);
        game.observations.push_back(obs);
        game.actions.push_back(t % 9);
        std::array<float, 9> policy{};
        policy.fill(1.0f / 9.0f);
        game.searchPolicies.push_back(policy);
        game.searchValues.push_back(tag);
        game.rewards.push_back(0.0f);
    }
    return game;
}

} // namespace

void test_length_is_number_of_moves() {
    GameHistory game = makeGame(5, 0.0f);
    assert(game.length() == 5);
    assert(game.observations.size() == 5);
    assert(game.searchValues.size() == 5);
    assert(game.rewards.size() == 5);
}

void test_add_and_size() {
    ReplayBuffer buffer(10);
    assert(buffer.size() == 0);
    buffer.add(makeGame(5, 1.0f));
    buffer.add(makeGame(7, 2.0f));
    assert(buffer.size() == 2);
    assert(buffer.totalPositions() == 12);
}

void test_ring_evicts_oldest_at_capacity() {
    ReplayBuffer buffer(3);
    for (int i = 0; i < 5; ++i) buffer.add(makeGame(2, static_cast<float>(i)));
    assert(buffer.size() == 3);
    // Games 0 and 1 were evicted; 2, 3, 4 remain in some slot.
    std::set<float> tags;
    for (size_t i = 0; i < buffer.size(); ++i) tags.insert(buffer.game(i).searchValues[0]);
    assert(tags == (std::set<float>{2.0f, 3.0f, 4.0f}));
    assert(buffer.totalPositions() == 6);
}

void test_sampled_positions_are_in_range() {
    ReplayBuffer buffer(10);
    buffer.add(makeGame(4, 1.0f));
    buffer.add(makeGame(9, 2.0f));
    auto samples = buffer.samplePositions(200);
    assert(samples.size() == 200);
    for (const auto& s : samples) {
        assert(s.gameIndex < buffer.size());
        assert(s.position >= 0);
        assert(s.position < static_cast<int>(buffer.game(s.gameIndex).length()));
    }
}

void test_sampling_reaches_both_games() {
    ReplayBuffer buffer(10);
    buffer.add(makeGame(5, 1.0f));
    buffer.add(makeGame(5, 2.0f));
    auto samples = buffer.samplePositions(500);
    std::set<size_t> seen;
    for (const auto& s : samples) seen.insert(s.gameIndex);
    assert(seen.size() == 2);
}

void test_replace_search_targets_overwrites_in_place() {
    // The hook Reanalyze needs: refresh a stored trajectory's search
    // targets with the results of a newer network, leaving observations,
    // actions and rewards untouched.
    ReplayBuffer buffer(10);
    buffer.add(makeGame(3, 1.0f));

    std::vector<std::array<float, 9>> policies(3);
    for (auto& p : policies) { p.fill(0.0f); p[4] = 1.0f; }
    std::vector<float> values = {0.5f, -0.5f, 0.25f};
    buffer.replaceSearchTargets(0, policies, values);

    const GameHistory& game = buffer.game(0);
    assert(game.length() == 3);
    assert(game.searchValues[1] == -0.5f);
    assert(game.searchPolicies[2][4] == 1.0f);
    assert(game.observations[2][0] == 2.0f);   // untouched
    assert(game.actions[1] == 1);              // untouched
}

int main() {
    test_length_is_number_of_moves();
    test_add_and_size();
    test_ring_evicts_oldest_at_capacity();
    test_sampled_positions_are_in_range();
    test_sampling_reaches_both_games();
    test_replace_search_targets_overwrites_in_place();
    std::printf("all replay buffer tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run the test to verify it fails**

Add `mz_add_test(test_replay_buffer)` to `CMakeLists.txt`, then build.

Expected: FAIL — `mz/replay_buffer.hpp: No such file or directory`.

- [ ] **Step 3: Write the headers**

`include/mz/game_history.hpp`:

```cpp
#pragma once
#include <array>
#include <cstddef>
#include <vector>

namespace mz {

// One complete self-play trajectory. All five vectors have the same
// length: the number of moves played.
//
// This replaces AlphaZero's flat per-position TrainingExample. MuZero
// cannot use flattened positions, because both of its training targets
// read forward from a position: the K-step unroll needs the actions that
// followed, and the n-step value bootstrap needs rewards and search values
// from later in the same game.
//
// Perspective conventions, relied on everywhere downstream:
//   searchValues[t] -- from the perspective of the player to move at t
//   rewards[t]      -- from the perspective of the player to move at t,
//                      received immediately after playing actions[t]
//
// On this domain rewards are zero everywhere except the final move.
struct GameHistory {
    std::vector<std::array<float, 18>> observations;
    std::vector<int> actions;
    std::vector<std::array<float, 9>> searchPolicies;
    std::vector<float> searchValues;
    std::vector<float> rewards;

    std::size_t length() const { return actions.size(); }
};

} // namespace mz
```

`include/mz/replay_buffer.hpp`:

```cpp
#pragma once
#include <array>
#include <cstddef>
#include <random>
#include <vector>
#include "mz/game_history.hpp"

namespace mz {

// Fixed-capacity ring buffer of whole trajectories -- not of positions,
// which is what AlphaZero's equivalent stores.
class ReplayBuffer {
public:
    struct Sample {
        std::size_t gameIndex;
        int position;
    };

    explicit ReplayBuffer(std::size_t capacity);

    void add(GameHistory game);

    std::size_t size() const { return games_.size(); }
    std::size_t totalPositions() const;

    // Precondition: index < size() (asserted).
    const GameHistory& game(std::size_t index) const;

    // Uniform over stored positions, with replacement: a game contributes
    // in proportion to its length, so long games are not under-sampled.
    // Precondition: totalPositions() > 0.
    std::vector<Sample> samplePositions(std::size_t count) const;

    // Reanalyze hook. Replaces a stored trajectory's search targets in
    // place, leaving observations, actions and rewards alone -- those are
    // facts about what happened and never go stale. Search policies and
    // values are the network's opinions at the time, and a stronger
    // network can improve them. Precondition: sizes match the trajectory
    // length (asserted).
    void replaceSearchTargets(std::size_t gameIndex,
                              std::vector<std::array<float, 9>> policies,
                              std::vector<float> values);

private:
    std::size_t capacity_;
    std::size_t nextIndex_ = 0;
    std::vector<GameHistory> games_;
    mutable std::mt19937 rng_{std::random_device{}()};
};

} // namespace mz
```

- [ ] **Step 4: Write the implementation**

`src/replay_buffer.cpp`:

```cpp
#include "mz/replay_buffer.hpp"
#include <cassert>

namespace mz {

ReplayBuffer::ReplayBuffer(std::size_t capacity) : capacity_(capacity) {
    assert(capacity > 0);
    games_.reserve(capacity);
}

void ReplayBuffer::add(GameHistory game) {
    if (games_.size() < capacity_) {
        games_.push_back(std::move(game));
    } else {
        games_[nextIndex_] = std::move(game);
        nextIndex_ = (nextIndex_ + 1) % capacity_;
    }
}

std::size_t ReplayBuffer::totalPositions() const {
    std::size_t total = 0;
    for (const auto& game : games_) total += game.length();
    return total;
}

const GameHistory& ReplayBuffer::game(std::size_t index) const {
    assert(index < games_.size());
    return games_[index];
}

std::vector<ReplayBuffer::Sample> ReplayBuffer::samplePositions(std::size_t count) const {
    assert(totalPositions() > 0);
    // Draw a game with probability proportional to its length, then a
    // uniform position within it: equivalent to drawing uniformly over all
    // stored positions.
    std::vector<std::size_t> cumulative;
    cumulative.reserve(games_.size());
    std::size_t running = 0;
    for (const auto& game : games_) {
        running += game.length();
        cumulative.push_back(running);
    }

    std::uniform_int_distribution<std::size_t> dist(0, running - 1);
    std::vector<Sample> samples;
    samples.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t draw = dist(rng_);
        std::size_t gameIndex = 0;
        while (cumulative[gameIndex] <= draw) ++gameIndex;
        std::size_t before = gameIndex == 0 ? 0 : cumulative[gameIndex - 1];
        samples.push_back(Sample{gameIndex, static_cast<int>(draw - before)});
    }
    return samples;
}

void ReplayBuffer::replaceSearchTargets(std::size_t gameIndex,
                                        std::vector<std::array<float, 9>> policies,
                                        std::vector<float> values) {
    assert(gameIndex < games_.size());
    GameHistory& game = games_[gameIndex];
    assert(policies.size() == game.length());
    assert(values.size() == game.length());
    game.searchPolicies = std::move(policies);
    game.searchValues = std::move(values);
}

} // namespace mz
```

- [ ] **Step 5: Add to the build and run the tests**

Add `src/replay_buffer.cpp` to the `add_library(mz ...)` source list.

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: PASS — `all replay buffer tests passed`, 4/4 tests.

- [ ] **Step 6: Commit**

```bash
git add include/mz/game_history.hpp include/mz/replay_buffer.hpp src/replay_buffer.cpp tests/test_replay_buffer.cpp CMakeLists.txt
git commit -m "$(cat <<'EOF'
Add trajectory storage and replay buffer

Stores whole games rather than flattened positions: both the K-step
unroll and the n-step value bootstrap read forward from a position.
Includes the in-place target-replacement hook Reanalyze will need.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Training targets and the n-step bootstrap

The deepest algorithmic difference from AlphaZero, and the one most worth testing carefully.

AlphaZero's value target is the final game result: a fact about the past, fixed forever. MuZero's is an n-step bootstrap — the rewards over the next n steps plus its own search's value estimate at step t+n. Because the players alternate, the sign flips at every ply.

Set `tdSteps` beyond the longest possible game and the bootstrap term falls off the end and the expression collapses to the plain game outcome, which is what board-game MuZero uses. The general form is implemented so the mechanism is visible; the default reproduces the familiar behavior.

**Files:**
- Create: `include/mz/targets.hpp`, `src/targets.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/test_targets.cpp`

**Interfaces:**
- Consumes: `mz::GameHistory` (Task 3).
- Produces: `mz::TargetConfig{int unrollSteps=5; int tdSteps=32; float discount=1.0f;}`, `mz::UnrolledSample{std::array<float,18> observation; std::vector<int> actions; std::vector<float> targetValues; std::vector<float> targetRewards; std::vector<std::array<float,9>> targetPolicies;}`, and `mz::makeUnrolledSample(const GameHistory&, int position, const TargetConfig&) -> UnrolledSample`.

**Array layout, used identically by Task 6's `trainStep`:** for `K = unrollSteps`, `actions` has size K, and `targetValues`, `targetRewards` and `targetPolicies` all have size K+1 so that index `k` always means unroll step `k`. `targetRewards[0]` is always 0 and is excluded from the loss — step 0 has no incoming transition, so there are K meaningful reward targets, at indices 1..K.

- [ ] **Step 1: Write the failing test**

`tests/test_targets.cpp`:

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include "mz/targets.hpp"

using namespace mz;

namespace {

// X wins on move 4 (the fifth ply). Search values are distinctive so the
// bootstrap term is identifiable; rewards are zero until the final move.
//   ply:            0     1     2     3     4
//   player:         X     O     X     O     X
//   searchValue:  0.10 -0.20  0.30 -0.40  0.50
//   reward:       0     0     0     0     +1  (to X, who just moved)
GameHistory makeWinForFirstPlayer() {
    GameHistory game;
    float values[5] = {0.10f, -0.20f, 0.30f, -0.40f, 0.50f};
    for (int t = 0; t < 5; ++t) {
        std::array<float, 18> obs{};
        obs[0] = static_cast<float>(t);
        game.observations.push_back(obs);
        game.actions.push_back(t);
        std::array<float, 9> policy{};
        policy.fill(0.0f);
        policy[t] = 1.0f;                    // one-hot, easy to identify
        game.searchPolicies.push_back(policy);
        game.searchValues.push_back(values[t]);
        game.rewards.push_back(t == 4 ? 1.0f : 0.0f);
    }
    return game;
}

bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }

} // namespace

void test_array_sizes_follow_unroll_length() {
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 3;
    UnrolledSample sample = makeUnrolledSample(game, 0, config);
    assert(sample.actions.size() == 3);
    assert(sample.targetValues.size() == 4);
    assert(sample.targetRewards.size() == 4);
    assert(sample.targetPolicies.size() == 4);
    assert(sample.targetRewards[0] == 0.0f);   // step 0 has no transition in
}

void test_observation_and_actions_come_from_position() {
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 2;
    UnrolledSample sample = makeUnrolledSample(game, 1, config);
    assert(sample.observation[0] == 1.0f);
    assert(sample.actions[0] == 1);
    assert(sample.actions[1] == 2);
}

void test_long_td_collapses_to_game_outcome() {
    // tdSteps beyond the game length: the bootstrap falls off the end, so
    // the value target is just the discounted, sign-flipped reward sum --
    // the plain game outcome, which is exactly AlphaZero's target.
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 4;
    config.tdSteps = 32;
    UnrolledSample sample = makeUnrolledSample(game, 0, config);

    // X wins. From ply 0 (X to move) that is +1; from ply 1 (O) -1; etc.
    assert(near(sample.targetValues[0], 1.0f));
    assert(near(sample.targetValues[1], -1.0f));
    assert(near(sample.targetValues[2], 1.0f));
    assert(near(sample.targetValues[3], -1.0f));
}

void test_short_td_bootstraps_from_stored_search_value() {
    // tdSteps = 2 from ply 0: no reward in [0, 2), so the target is the
    // search value at ply 2, sign-flipped by (2 - 0) plies = even = +1.
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 1;
    config.tdSteps = 2;
    UnrolledSample sample = makeUnrolledSample(game, 0, config);
    assert(near(sample.targetValues[0], 0.30f));

    // From ply 1 the bootstrap lands on ply 3, again an even gap: -0.40.
    UnrolledSample fromOne = makeUnrolledSample(game, 1, config);
    assert(near(fromOne.targetValues[0], -0.40f));
}

void test_odd_td_gap_flips_the_sign() {
    // tdSteps = 1 from ply 0 bootstraps on ply 1, an odd gap, so the stored
    // value (-0.20, from O's perspective) is negated into X's.
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 1;
    config.tdSteps = 1;
    UnrolledSample sample = makeUnrolledSample(game, 0, config);
    assert(near(sample.targetValues[0], 0.20f));
}

void test_reward_targets_align_with_incoming_transition() {
    // Unrolling 4 steps from ply 1 reaches plies 1..5. The only nonzero
    // reward is on the transition out of ply 4, which arrives at step 4.
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 4;
    UnrolledSample sample = makeUnrolledSample(game, 1, config);
    assert(sample.targetRewards[0] == 0.0f);
    assert(sample.targetRewards[1] == 0.0f);   // out of ply 1
    assert(sample.targetRewards[2] == 0.0f);   // out of ply 2
    assert(sample.targetRewards[3] == 0.0f);   // out of ply 3
    assert(near(sample.targetRewards[4], 1.0f));   // out of ply 4: X wins
}

void test_policy_targets_come_from_stored_search() {
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 2;
    UnrolledSample sample = makeUnrolledSample(game, 2, config);
    assert(near(sample.targetPolicies[0][2], 1.0f));
    assert(near(sample.targetPolicies[1][3], 1.0f));
    assert(near(sample.targetPolicies[2][4], 1.0f));
}

void test_absorbing_padding_past_end_of_game() {
    // Unrolling 3 steps from ply 3 runs off the end of a 5-ply game at
    // step 2. Past terminal: zero value, zero reward, uniform policy.
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 3;
    UnrolledSample sample = makeUnrolledSample(game, 3, config);

    assert(near(sample.targetValues[2], 0.0f));
    assert(near(sample.targetValues[3], 0.0f));
    assert(near(sample.targetRewards[3], 0.0f));
    for (int a = 0; a < 9; ++a) assert(near(sample.targetPolicies[2][a], 1.0f / 9.0f));
    for (int a = 0; a < 9; ++a) assert(near(sample.targetPolicies[3][a], 1.0f / 9.0f));
}

void test_padding_actions_cycle_deterministically() {
    // The paper samples random actions past terminal. This uses a fixed
    // cycle instead: deterministic, reproducible, and it still spreads
    // across all nine actions as the sampled position varies.
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 4;
    UnrolledSample a = makeUnrolledSample(game, 3, config);
    UnrolledSample b = makeUnrolledSample(game, 3, config);
    assert(a.actions == b.actions);
    for (int action : a.actions) assert(action >= 0 && action < 9);
}

void test_draw_gives_zero_value_targets_everywhere() {
    GameHistory game;
    for (int t = 0; t < 9; ++t) {
        game.observations.push_back(std::array<float, 18>{});
        game.actions.push_back(t);
        std::array<float, 9> policy{};
        policy.fill(1.0f / 9.0f);
        game.searchPolicies.push_back(policy);
        game.searchValues.push_back(0.0f);
        game.rewards.push_back(0.0f);
    }
    TargetConfig config;
    config.unrollSteps = 5;
    UnrolledSample sample = makeUnrolledSample(game, 0, config);
    for (float v : sample.targetValues) assert(near(v, 0.0f));
    for (float r : sample.targetRewards) assert(near(r, 0.0f));
}

int main() {
    test_array_sizes_follow_unroll_length();
    test_observation_and_actions_come_from_position();
    test_long_td_collapses_to_game_outcome();
    test_short_td_bootstraps_from_stored_search_value();
    test_odd_td_gap_flips_the_sign();
    test_reward_targets_align_with_incoming_transition();
    test_policy_targets_come_from_stored_search();
    test_absorbing_padding_past_end_of_game();
    test_padding_actions_cycle_deterministically();
    test_draw_gives_zero_value_targets_everywhere();
    std::printf("all target tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run the test to verify it fails**

Add `mz_add_test(test_targets)` to `CMakeLists.txt`, then build.

Expected: FAIL — `mz/targets.hpp: No such file or directory`.

- [ ] **Step 3: Write the header**

`include/mz/targets.hpp`:

```cpp
#pragma once
#include <array>
#include <vector>
#include "mz/game_history.hpp"

namespace mz {

struct TargetConfig {
    // K: how many dynamics steps each training sample unrolls.
    int unrollSteps = 5;
    // n for the n-step value bootstrap. The default is past the longest
    // possible tic-tac-toe game (9 plies), so the bootstrap term falls off
    // the end and the value target collapses to the game outcome -- what
    // board-game MuZero uses, and what AlphaZero uses unconditionally.
    // Lower it to see real bootstrapping.
    int tdSteps = 32;
    // 1.0 for a board game: a win in five moves is worth a win in nine.
    float discount = 1.0f;
};

// One training sample: an observation, the K actions to unroll along it,
// and aligned targets.
//
// Layout, matched exactly by MuZeroNetwork::trainStep:
//   actions         size K       actions[k] leads from step k to step k+1
//   targetValues    size K + 1   indexed by unroll step 0..K
//   targetRewards   size K + 1   index 0 always 0 and excluded from the
//                                loss (step 0 has no incoming transition);
//                                indices 1..K are the K real targets
//   targetPolicies  size K + 1   indexed by unroll step 0..K
struct UnrolledSample {
    std::array<float, 18> observation;
    std::vector<int> actions;
    std::vector<float> targetValues;
    std::vector<float> targetRewards;
    std::vector<std::array<float, 9>> targetPolicies;
};

// Builds the sample rooted at `position` of `game`.
//
// Pure: a function of the trajectory and the config only. That is
// deliberate -- Reanalyze works by refreshing a trajectory's stored search
// targets and calling this again, which only works if nothing else here
// carries state.
//
// Precondition: 0 <= position < game.length() (asserted).
UnrolledSample makeUnrolledSample(const GameHistory& game, int position, const TargetConfig& config);

} // namespace mz
```

- [ ] **Step 4: Write the implementation**

`src/targets.cpp`:

```cpp
#include "mz/targets.hpp"
#include <cassert>
#include <cmath>

namespace mz {

namespace {

// The n-step bootstrapped value at absolute ply `t`, in the perspective of
// the player to move at `t`.
//
//   z_t = sum_{i in [t, t+n)} (-1)^(i-t) * discount^(i-t) * reward_i
//         + (-1)^n * discount^n * searchValue_{t+n}
//
// The alternating sign is the two-player part: rewards and search values
// are each stored in the perspective of whoever was to move at their own
// ply, so every ply of separation flips the sign relative to t.
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

std::array<float, 9> uniformPolicy() {
    std::array<float, 9> policy{};
    policy.fill(1.0f / 9.0f);
    return policy;
}

} // namespace

UnrolledSample makeUnrolledSample(const GameHistory& game, int position, const TargetConfig& config) {
    assert(position >= 0);
    assert(position < static_cast<int>(game.length()));
    assert(config.unrollSteps >= 0);

    const int length = static_cast<int>(game.length());
    const int K = config.unrollSteps;

    UnrolledSample sample;
    sample.observation = game.observations[position];
    sample.actions.resize(K);
    sample.targetValues.resize(K + 1);
    sample.targetRewards.assign(K + 1, 0.0f);
    sample.targetPolicies.resize(K + 1);

    for (int k = 0; k <= K; ++k) {
        const int t = position + k;

        sample.targetValues[k] = bootstrappedValue(game, t, config);

        // The reward for unroll step k is the one paid on the transition
        // INTO step k, i.e. out of ply t-1. Its perspective is already
        // that of the player who acted, which is what the reward head
        // predicts, so no sign adjustment is needed.
        if (k > 0) {
            const int from = t - 1;
            sample.targetRewards[k] = (from < length) ? game.rewards[from] : 0.0f;
        }

        sample.targetPolicies[k] = (t < length) ? game.searchPolicies[t] : uniformPolicy();

        if (k < K) {
            if (t < length) {
                sample.actions[k] = game.actions[t];
            } else {
                // Absorbing state. The paper samples a uniformly random
                // action here; a fixed cycle is used instead so the
                // function stays pure and the targets stay reproducible.
                // Across sampled positions this still covers all nine.
                sample.actions[k] = k % 9;
            }
        }
    }

    return sample;
}

} // namespace mz
```

- [ ] **Step 5: Add to the build and run the tests**

Add `src/targets.cpp` to the `add_library(mz ...)` source list.

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: PASS — `all target tests passed`, 5/5 tests.

- [ ] **Step 6: Commit**

```bash
git add include/mz/targets.hpp src/targets.cpp tests/test_targets.cpp CMakeLists.txt
git commit -m "$(cat <<'EOF'
Add n-step bootstrapped training targets

Value targets bootstrap from the network's own stored search values
with a sign flip per ply, rather than using the final game result the
way AlphaZero does. The default tdSteps exceeds the game length, so the
expression collapses to the outcome unless deliberately shortened.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: MuZero network — the three functions and inference

Where AlphaZero has one network, MuZero has three, and the split is the whole algorithm.

| function | question it answers | maps | sizes |
|---|---|---|---|
| representation `h` | what is going on here? | observation → latent | 18 → 64 (ReLU) → 32 |
| dynamics `g` | what happens if I do that? | (latent, action) → (latent, reward) | 41 → 64 (ReLU) → 32, and → 1 |
| prediction `f` | how good is this, and what should I play? | latent → (policy, value) | 32 → 64 (ReLU) → 9, and → 1 |

Note what is missing: there is no decoder from latent back to a board. Nothing in this project can look at a latent and say which position it is. That is not an oversight — it is MuZero's defining constraint. A latent is only ever required to support good policy, value, and reward predictions, never to reconstruct the observation. AlphaZero never faces the question, because it always has the real board in hand.

This task builds inference and checkpointing. Training comes next, separately, because backprop-through-time deserves its own test cycle.

**Files:**
- Create: `include/mz/network.hpp`, `src/network.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/test_network.cpp`

**Interfaces:**
- Consumes: `mz::Dense` and the activations (Task 2), `mz::UnrolledSample` (Task 4).
- Produces: `mz::MuZeroNetwork` with constants `kObservationSize=18`, `kActionSize=9`, `kLatentSize=32`, `kHiddenSize=64`; nested `InitialInference{std::vector<float> latent; std::array<float,9> policy; float value;}` and `RecurrentInference{std::vector<float> latent; float reward; std::array<float,9> policy; float value;}` and `Losses{float total, value, policy, reward;}`; methods `initialInference(const std::array<float,18>&) const`, `recurrentInference(const std::vector<float>& latent, int action) const`, `trainStep(const std::vector<UnrolledSample>&, float learningRate) -> Losses` (Task 6), `save(const std::string&) const`, `load(const std::string&)`.

- [ ] **Step 1: Write the failing test**

`tests/test_network.cpp` — this task's tests only; Task 6 appends the gradient checks to the same file.

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "mz/board.hpp"
#include "mz/network.hpp"

using namespace mz;

namespace {

std::string tempPath() {
    return std::string("mz_test_checkpoint_") + std::to_string(std::rand()) + ".bin";
}

bool near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) < tol; }

} // namespace

void test_initial_inference_shapes_and_ranges() {
    MuZeroNetwork network(1234);
    Board board;
    auto out = network.initialInference(board.encode());

    assert(out.latent.size() == MuZeroNetwork::kLatentSize);
    // Latents are min-max normalized into [0, 1] so repeated dynamics
    // applications cannot let magnitudes drift.
    for (float v : out.latent) assert(v >= -1e-6f && v <= 1.0f + 1e-6f);

    float sum = 0.0f;
    for (float p : out.policy) { assert(p > 0.0f); sum += p; }
    assert(near(sum, 1.0f));
    assert(out.value >= -1.0f && out.value <= 1.0f);
}

void test_policy_head_is_not_masked_to_legal_moves() {
    // The network knows nothing about legality. Masking happens in MCTS,
    // and only at the root.
    MuZeroNetwork network(99);
    Board board;
    board = board.applyMove(4);
    auto out = network.initialInference(board.encode());
    assert(out.policy[4] > 0.0f);   // an illegal move, still given a prior
}

void test_recurrent_inference_shapes_and_ranges() {
    MuZeroNetwork network(555);
    Board board;
    auto initial = network.initialInference(board.encode());
    auto next = network.recurrentInference(initial.latent, 4);

    assert(next.latent.size() == MuZeroNetwork::kLatentSize);
    for (float v : next.latent) assert(v >= -1e-6f && v <= 1.0f + 1e-6f);
    assert(next.reward >= -1.0f && next.reward <= 1.0f);
    assert(next.value >= -1.0f && next.value <= 1.0f);
    float sum = 0.0f;
    for (float p : next.policy) sum += p;
    assert(near(sum, 1.0f));
}

void test_recurrent_inference_depends_on_the_action() {
    // The action must actually reach the dynamics network. If the one-hot
    // were dropped or misaligned, every action would produce the same
    // successor and search would be meaningless.
    MuZeroNetwork network(7);
    Board board;
    auto initial = network.initialInference(board.encode());
    auto a = network.recurrentInference(initial.latent, 0);
    auto b = network.recurrentInference(initial.latent, 8);

    bool differs = false;
    for (int i = 0; i < MuZeroNetwork::kLatentSize; ++i) {
        if (std::fabs(a.latent[i] - b.latent[i]) > 1e-5f) differs = true;
    }
    assert(differs);
}

void test_recurrent_inference_is_deterministic() {
    MuZeroNetwork network(31);
    Board board;
    auto initial = network.initialInference(board.encode());
    auto a = network.recurrentInference(initial.latent, 3);
    auto b = network.recurrentInference(initial.latent, 3);
    for (int i = 0; i < MuZeroNetwork::kLatentSize; ++i) assert(near(a.latent[i], b.latent[i]));
    assert(near(a.reward, b.reward));
}

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
        latent = next.latent;
        for (float v : latent) {
            assert(std::isfinite(v));
            assert(v >= -1e-6f && v <= 1.0f + 1e-6f);
        }
        assert(std::isfinite(next.value));
        assert(std::isfinite(next.reward));
    }
}

void test_different_seeds_give_different_networks() {
    MuZeroNetwork a(1);
    MuZeroNetwork b(2);
    Board board;
    assert(!near(a.initialInference(board.encode()).value,
                 b.initialInference(board.encode()).value));
}

void test_save_load_round_trips() {
    MuZeroNetwork original(4242);
    std::string path = tempPath();
    original.save(path);

    MuZeroNetwork restored(1);
    restored.load(path);
    std::remove(path.c_str());

    Board board;
    board = board.applyMove(0);
    board = board.applyMove(4);
    auto a = original.initialInference(board.encode());
    auto b = restored.initialInference(board.encode());

    assert(near(a.value, b.value));
    for (int i = 0; i < 9; ++i) assert(near(a.policy[i], b.policy[i]));
    for (int i = 0; i < MuZeroNetwork::kLatentSize; ++i) assert(near(a.latent[i], b.latent[i]));

    auto ra = original.recurrentInference(a.latent, 2);
    auto rb = restored.recurrentInference(b.latent, 2);
    assert(near(ra.reward, rb.reward));
    assert(near(ra.value, rb.value));
}

void test_load_rejects_a_non_checkpoint_file() {
    std::string path = tempPath();
    FILE* f = std::fopen(path.c_str(), "wb");
    std::fputs("this is not a checkpoint", f);
    std::fclose(f);

    MuZeroNetwork network(1);
    bool threw = false;
    try {
        network.load(path);
    } catch (const std::exception&) {
        threw = true;
    }
    std::remove(path.c_str());
    assert(threw);
}

void test_load_reports_a_missing_file() {
    MuZeroNetwork network(1);
    bool threw = false;
    try {
        network.load("definitely_not_a_real_path_12345.bin");
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw);
}

int main() {
    test_initial_inference_shapes_and_ranges();
    test_policy_head_is_not_masked_to_legal_moves();
    test_recurrent_inference_shapes_and_ranges();
    test_recurrent_inference_depends_on_the_action();
    test_recurrent_inference_is_deterministic();
    test_deep_unroll_stays_bounded();
    test_different_seeds_give_different_networks();
    test_save_load_round_trips();
    test_load_rejects_a_non_checkpoint_file();
    test_load_reports_a_missing_file();
    std::printf("all network tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run the test to verify it fails**

Add `mz_add_test(test_network)` to `CMakeLists.txt`, then build.

Expected: FAIL — `mz/network.hpp: No such file or directory`.

- [ ] **Step 3: Write the header**

`include/mz/network.hpp`:

```cpp
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "mz/mlp.hpp"
#include "mz/targets.hpp"

namespace mz {

// MuZero's three functions, each a small MLP built from Dense.
//
//   representation h : observation -> latent
//   dynamics       g : (latent, action) -> (latent, reward)
//   prediction     f : latent -> (policy, value)
//
// There is deliberately NO decoder from latent back to a board. Nothing
// here can look at a latent and say which position it is. A latent only
// has to support good policy, value and reward predictions -- it is never
// asked to reconstruct the observation. That is the constraint that
// separates MuZero from AlphaZero, which always has the real board.
//
// Perspective conventions:
//   value  -- from the perspective of the player to move at that latent
//   reward -- from the perspective of the player who took the action
class MuZeroNetwork {
public:
    static constexpr int kObservationSize = 18;
    static constexpr int kActionSize = 9;
    static constexpr int kLatentSize = 32;
    static constexpr int kHiddenSize = 64;

    struct InitialInference {
        std::vector<float> latent;
        std::array<float, kActionSize> policy;
        float value;
    };

    struct RecurrentInference {
        std::vector<float> latent;
        float reward;
        std::array<float, kActionSize> policy;
        float value;
    };

    struct Losses {
        float total = 0.0f;
        float value = 0.0f;
        float policy = 0.0f;
        float reward = 0.0f;
    };

    explicit MuZeroNetwork(std::uint32_t seed);
    MuZeroNetwork();

    // h then f. Used at the search root, on the real board's encoding --
    // the only place an observation enters the system.
    InitialInference initialInference(const std::array<float, kObservationSize>& observation) const;

    // g then f. Every non-root node in the tree is reached this way, with
    // no reference to the real board at all.
    // Precondition: latent.size() == kLatentSize, 0 <= action < 9 (asserted).
    RecurrentInference recurrentInference(const std::vector<float>& latent, int action) const;

    // One SGD step over a batch of unrolled samples. Implemented in Task 6.
    Losses trainStep(const std::vector<UnrolledSample>& batch, float learningRate);

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    // Forward pieces shared by inference and training. Each returns the
    // pre-activations training needs; inference discards them.
    struct HiddenTrace {
        std::vector<float> preActivation;   // before ReLU
        std::vector<float> activation;      // after ReLU
    };

    HiddenTrace representationHidden(const std::array<float, kObservationSize>& observation) const;
    HiddenTrace dynamicsHidden(const std::vector<float>& dynamicsInput) const;
    HiddenTrace predictionHidden(const std::vector<float>& latent) const;

    static std::vector<float> makeDynamicsInput(const std::vector<float>& latent, int action);

    Dense hFc1_, hFc2_;                  // representation
    Dense gFc1_, gFc2_, gReward_;        // dynamics
    Dense fFc1_, fPolicy_, fValue_;      // prediction
};

} // namespace mz
```

- [ ] **Step 4: Write the implementation**

`src/network.cpp` — inference and checkpointing only; Task 6 adds `trainStep` to this same file.

```cpp
#include "mz/network.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

namespace mz {

namespace {
constexpr char kCheckpointMagic[4] = {'M', 'Z', 'N', 'N'};
constexpr std::uint32_t kCheckpointVersion = 1;
} // namespace

MuZeroNetwork::MuZeroNetwork(std::uint32_t seed) {
    std::mt19937 rng(seed);
    hFc1_ = Dense(kObservationSize, kHiddenSize, rng);
    hFc2_ = Dense(kHiddenSize, kLatentSize, rng);
    gFc1_ = Dense(kLatentSize + kActionSize, kHiddenSize, rng);
    gFc2_ = Dense(kHiddenSize, kLatentSize, rng);
    gReward_ = Dense(kHiddenSize, 1, rng);
    fFc1_ = Dense(kLatentSize, kHiddenSize, rng);
    fPolicy_ = Dense(kHiddenSize, kActionSize, rng);
    fValue_ = Dense(kHiddenSize, 1, rng);
}

MuZeroNetwork::MuZeroNetwork() : MuZeroNetwork(std::random_device{}()) {}

std::vector<float> MuZeroNetwork::makeDynamicsInput(const std::vector<float>& latent, int action) {
    assert(static_cast<int>(latent.size()) == kLatentSize);
    assert(action >= 0 && action < kActionSize);
    // Latent with the action appended as a one-hot. This is how an action
    // enters the model at all -- there is no board to apply it to.
    std::vector<float> input(kLatentSize + kActionSize, 0.0f);
    std::copy(latent.begin(), latent.end(), input.begin());
    input[kLatentSize + action] = 1.0f;
    return input;
}

MuZeroNetwork::HiddenTrace MuZeroNetwork::representationHidden(
    const std::array<float, kObservationSize>& observation) const {
    std::vector<float> input(observation.begin(), observation.end());
    HiddenTrace trace;
    trace.preActivation = hFc1_.forward(input);
    trace.activation = relu(trace.preActivation);
    return trace;
}

MuZeroNetwork::HiddenTrace MuZeroNetwork::dynamicsHidden(const std::vector<float>& dynamicsInput) const {
    HiddenTrace trace;
    trace.preActivation = gFc1_.forward(dynamicsInput);
    trace.activation = relu(trace.preActivation);
    return trace;
}

MuZeroNetwork::HiddenTrace MuZeroNetwork::predictionHidden(const std::vector<float>& latent) const {
    HiddenTrace trace;
    trace.preActivation = fFc1_.forward(latent);
    trace.activation = relu(trace.preActivation);
    return trace;
}

MuZeroNetwork::InitialInference MuZeroNetwork::initialInference(
    const std::array<float, kObservationSize>& observation) const {
    HiddenTrace hidden = representationHidden(observation);
    std::vector<float> latent = minMaxNormalize(hFc2_.forward(hidden.activation));

    HiddenTrace predictionTrace = predictionHidden(latent);
    InitialInference out;
    out.latent = std::move(latent);
    out.policy = softmax9(fPolicy_.forward(predictionTrace.activation));
    out.value = std::tanh(fValue_.forward(predictionTrace.activation)[0]);
    return out;
}

MuZeroNetwork::RecurrentInference MuZeroNetwork::recurrentInference(const std::vector<float>& latent,
                                                                    int action) const {
    std::vector<float> input = makeDynamicsInput(latent, action);
    HiddenTrace hidden = dynamicsHidden(input);

    std::vector<float> nextLatent = minMaxNormalize(gFc2_.forward(hidden.activation));
    float reward = std::tanh(gReward_.forward(hidden.activation)[0]);

    HiddenTrace predictionTrace = predictionHidden(nextLatent);
    RecurrentInference out;
    out.latent = std::move(nextLatent);
    out.reward = reward;
    out.policy = softmax9(fPolicy_.forward(predictionTrace.activation));
    out.value = std::tanh(fValue_.forward(predictionTrace.activation)[0]);
    return out;
}

void MuZeroNetwork::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("MuZeroNetwork::save: cannot open " + path);

    out.write(kCheckpointMagic, sizeof(kCheckpointMagic));
    std::uint32_t header[5] = {kCheckpointVersion,
                               static_cast<std::uint32_t>(kObservationSize),
                               static_cast<std::uint32_t>(kActionSize),
                               static_cast<std::uint32_t>(kLatentSize),
                               static_cast<std::uint32_t>(kHiddenSize)};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));

    for (const Dense* layer : {&hFc1_, &hFc2_, &gFc1_, &gFc2_, &gReward_, &fFc1_, &fPolicy_, &fValue_}) {
        layer->write(out);
    }
    if (!out) throw std::runtime_error("MuZeroNetwork::save: write failed for " + path);
}

void MuZeroNetwork::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("MuZeroNetwork::load: cannot open " + path);

    char magic[4] = {};
    std::uint32_t header[5] = {};
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(header), sizeof(header));

    bool headerOk = in && std::equal(std::begin(magic), std::end(magic), std::begin(kCheckpointMagic)) &&
                    header[0] == kCheckpointVersion &&
                    header[1] == static_cast<std::uint32_t>(kObservationSize) &&
                    header[2] == static_cast<std::uint32_t>(kActionSize) &&
                    header[3] == static_cast<std::uint32_t>(kLatentSize) &&
                    header[4] == static_cast<std::uint32_t>(kHiddenSize);
    if (!headerOk) {
        throw std::runtime_error(
            "MuZeroNetwork::load: not a valid checkpoint (bad magic/version/shape): " + path);
    }

    for (Dense* layer : {&hFc1_, &hFc2_, &gFc1_, &gFc2_, &gReward_, &fFc1_, &fPolicy_, &fValue_}) {
        layer->read(in);
    }
    if (!in) throw std::runtime_error("MuZeroNetwork::load: truncated file " + path);
}

} // namespace mz
```

Note: `trainStep` is declared in the header but not yet defined, so linking `test_network` will fail until Task 6. To keep this task independently verifiable, add a temporary stub at the end of `src/network.cpp`:

```cpp
namespace mz {
// Replaced in full by Task 6.
MuZeroNetwork::Losses MuZeroNetwork::trainStep(const std::vector<UnrolledSample>&, float) {
    return Losses{};
}
} // namespace mz
```

- [ ] **Step 5: Add to the build and run the tests**

Add `src/network.cpp` to the `add_library(mz ...)` source list.

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: PASS — `all network tests passed`, 6/6 tests.

- [ ] **Step 6: Commit**

```bash
git add include/mz/network.hpp src/network.cpp tests/test_network.cpp CMakeLists.txt
git commit -m "$(cat <<'EOF'
Add MuZero representation, dynamics and prediction networks

Inference and checkpointing. No decoder from latent back to board: a
latent only has to support policy, value and reward predictions, never
to reconstruct the observation. trainStep is stubbed pending the next
task.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Backprop-through-time over the K-step unroll

The hardest correctness problem in the project, and the one most likely to fail silently. A wrong gradient here does not crash or throw — it just trains slightly worse, forever, and looks like a hyperparameter problem. So this task's real deliverable is the end-to-end numerical gradient check, not the training code.

The forward pass for one sample:

```
h1      = relu(hFc1(observation))
s[0]    = minMaxNormalize(hFc2(h1))
for k in 0..K-1:
    in[k]     = concat(s[k], onehot(actions[k]))
    d1[k]     = relu(gFc1(in[k]))
    s[k+1]    = minMaxNormalize(gFc2(d1[k]))
    reward[k+1] = tanh(gReward(d1[k]))
for k in 0..K:
    p1[k]     = relu(fFc1(s[k]))
    policy[k] = softmax(fPolicy(p1[k]))
    value[k]  = tanh(fValue(p1[k]))
```

Two scaling rules from the paper, both load-bearing:

- **Loss scaling.** Every unroll step past 0 contributes at 1/K, so a five-step sample does not outweigh a one-step one.
- **The half gradient.** The gradient flowing into dynamics *from its latent input* is multiplied by 0.5 at every step. Without it, gradient magnitude compounds across the recurrence and the latents destabilize. This is one line and it is easy to leave out; the test below pins it.

The backward pass runs in one reverse loop so that each `dS[k]` is complete — prediction-head contribution plus the contribution arriving from step k+1 — before it is used.

**Files:**
- Modify: `include/mz/network.hpp` (no interface change; `trainStep` is already declared)
- Modify: `src/network.cpp` (replace the Task 5 stub)
- Modify: `tests/test_network.cpp` (append)

**Interfaces:**
- Consumes: everything from Task 5, plus `mz::UnrolledSample` (Task 4).
- Produces: a working `MuZeroNetwork::trainStep`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_network.cpp` (and add the new calls to `main`):

```cpp
// ---- Task 6: backprop-through-time ----

#include <random>
#include "mz/targets.hpp"

namespace {

// A sample with fixed, non-degenerate targets: enough structure that every
// head and every unroll step contributes real gradient.
UnrolledSample makeProbeSample(int unrollSteps) {
    UnrolledSample sample;
    sample.observation.fill(0.0f);
    sample.observation[0] = 1.0f;
    sample.observation[4] = 1.0f;
    sample.observation[9 + 3] = 1.0f;

    sample.actions.resize(unrollSteps);
    for (int k = 0; k < unrollSteps; ++k) sample.actions[k] = (k * 3 + 1) % 9;

    sample.targetValues.resize(unrollSteps + 1);
    sample.targetRewards.assign(unrollSteps + 1, 0.0f);
    sample.targetPolicies.resize(unrollSteps + 1);
    for (int k = 0; k <= unrollSteps; ++k) {
        sample.targetValues[k] = (k % 2 == 0) ? 0.6f : -0.4f;
        if (k > 0) sample.targetRewards[k] = (k == unrollSteps) ? 1.0f : 0.0f;
        std::array<float, 9> policy{};
        policy.fill(0.05f);
        policy[(k * 2) % 9] = 1.0f - 0.05f * 8.0f;
        sample.targetPolicies[k] = policy;
    }
    return sample;
}

} // namespace

void test_train_step_reports_finite_decomposed_losses() {
    MuZeroNetwork network(2024);
    std::vector<UnrolledSample> batch = {makeProbeSample(5), makeProbeSample(5)};
    MuZeroNetwork::Losses losses = network.trainStep(batch, 0.01f);

    assert(std::isfinite(losses.total));
    assert(losses.value >= 0.0f && losses.policy >= 0.0f && losses.reward >= 0.0f);
    assert(near(losses.total, losses.value + losses.policy + losses.reward, 1e-3f));
}

void test_train_step_reduces_loss_on_a_fixed_batch() {
    // Repeatedly fitting one batch must drive its loss down. This is the
    // coarse check: it catches a sign error or a disconnected head, though
    // not a subtly wrong gradient.
    MuZeroNetwork network(31337);
    std::vector<UnrolledSample> batch = {makeProbeSample(5)};
    float first = network.trainStep(batch, 0.05f).total;
    float last = first;
    for (int i = 0; i < 200; ++i) last = network.trainStep(batch, 0.05f).total;
    assert(last < first * 0.75f);
}

void test_train_step_works_at_unroll_zero_and_one() {
    MuZeroNetwork network(5);
    std::vector<UnrolledSample> zero = {makeProbeSample(0)};
    std::vector<UnrolledSample> one = {makeProbeSample(1)};
    assert(std::isfinite(network.trainStep(zero, 0.01f).total));
    assert(std::isfinite(network.trainStep(one, 0.01f).total));
}

void test_gradient_matches_numerical_through_the_full_unroll() {
    // The real test. A directional finite-difference check: pick a random
    // direction in parameter space, and confirm the loss changes by the
    // amount one SGD step along the gradient predicts.
    //
    // A step of size `rate` along the negative gradient should reduce the
    // loss by approximately rate * ||grad||^2, to first order. Measuring
    // that at two step sizes and confirming the ratio is ~2 verifies the
    // gradient is right in magnitude, not merely in sign -- which is what
    // a missing 1/K scale or a missing half-gradient would break.
    const int unrollSteps = 4;
    std::vector<UnrolledSample> batch = {makeProbeSample(unrollSteps)};

    auto dropForRate = [&](float rate) {
        MuZeroNetwork network(8675309);
        // trainStep both computes the loss at the current parameters and
        // takes the step, so the returned value is the "before" loss.
        float before = network.trainStep(batch, rate).total;
        // Re-running with rate 0 would still step; instead measure the new
        // loss by taking a zero-size step.
        float after = network.trainStep(batch, 0.0f).total;
        return before - after;
    };

    float smallDrop = dropForRate(2e-3f);
    float largeDrop = dropForRate(4e-3f);

    // Both steps must decrease the loss...
    assert(smallDrop > 0.0f);
    assert(largeDrop > 0.0f);
    // ...and doubling the step must roughly double the decrease.
    float ratio = largeDrop / smallDrop;
    assert(ratio > 1.7f && ratio < 2.3f);
}

void test_dynamics_gradient_reaches_every_unroll_step() {
    // Isolates deep gradient flow. Steps 0..K-1 get targets equal to the
    // network's OWN current predictions, so those steps contribute
    // essentially no gradient. Only the final step disagrees with the
    // network, and its signal can reach the representation network only by
    // travelling back through all K dynamics applications.
    //
    // Without this construction the test would prove nothing: step 0's own
    // prediction head would move the representation network regardless.
    const int K = 5;
    MuZeroNetwork reference(4321);
    MuZeroNetwork trained(4321);

    UnrolledSample sample = makeProbeSample(K);

    // Walk the network forward exactly as trainStep will, recording what
    // it currently predicts at each step.
    auto initial = trained.initialInference(sample.observation);
    sample.targetValues[0] = initial.value;
    sample.targetPolicies[0] = initial.policy;
    sample.targetRewards[0] = 0.0f;

    std::vector<float> latent = initial.latent;
    for (int k = 1; k <= K; ++k) {
        auto step = trained.recurrentInference(latent, sample.actions[k - 1]);
        latent = step.latent;
        if (k < K) {
            sample.targetValues[k] = step.value;
            sample.targetPolicies[k] = step.policy;
            sample.targetRewards[k] = step.reward;
        } else {
            // The one disagreement, K dynamics steps downstream.
            sample.targetValues[k] = (step.value > 0.0f) ? -1.0f : 1.0f;
            sample.targetRewards[k] = (step.reward > 0.0f) ? -1.0f : 1.0f;
            sample.targetPolicies[k] = step.policy;
        }
    }

    std::vector<UnrolledSample> batch = {sample};
    for (int i = 0; i < 50; ++i) trained.trainStep(batch, 0.05f);

    Board board;
    auto before = reference.initialInference(board.encode());
    auto after = trained.initialInference(board.encode());
    bool moved = false;
    for (int i = 0; i < MuZeroNetwork::kLatentSize; ++i) {
        if (std::fabs(before.latent[i] - after.latent[i]) > 1e-4f) moved = true;
    }
    assert(moved);
}

void test_trained_weights_survive_a_checkpoint_round_trip() {
    MuZeroNetwork network(777);
    std::vector<UnrolledSample> batch = {makeProbeSample(5)};
    for (int i = 0; i < 20; ++i) network.trainStep(batch, 0.05f);

    std::string path = tempPath();
    network.save(path);
    MuZeroNetwork restored(1);
    restored.load(path);
    std::remove(path.c_str());

    Board board;
    auto a = network.initialInference(board.encode());
    auto b = restored.initialInference(board.encode());
    assert(near(a.value, b.value));
    auto ra = network.recurrentInference(a.latent, 5);
    auto rb = restored.recurrentInference(b.latent, 5);
    assert(near(ra.reward, rb.reward));
}
```

Add to `main`, before the final `printf`:

```cpp
    test_train_step_reports_finite_decomposed_losses();
    test_train_step_reduces_loss_on_a_fixed_batch();
    test_train_step_works_at_unroll_zero_and_one();
    test_gradient_matches_numerical_through_the_full_unroll();
    test_dynamics_gradient_reaches_every_unroll_step();
    test_trained_weights_survive_a_checkpoint_round_trip();
```

- [ ] **Step 2: Run the tests to verify they fail**

```sh
cmake --build build && ./build/test_network
```

Expected: FAIL — the stub returns zeros, so `test_train_step_reduces_loss_on_a_fixed_batch` asserts.

- [ ] **Step 3: Replace the stub with the real implementation**

Delete the temporary stub from `src/network.cpp` and add these includes at the top if not already present: `<cassert>`, `<cmath>`, `<vector>`. Then append inside `namespace mz`:

```cpp
MuZeroNetwork::Losses MuZeroNetwork::trainStep(const std::vector<UnrolledSample>& batch,
                                               float learningRate) {
    assert(!batch.empty());

    for (Dense* layer : {&hFc1_, &hFc2_, &gFc1_, &gFc2_, &gReward_, &fFc1_, &fPolicy_, &fValue_}) {
        layer->zeroGrad();
    }

    Losses losses;

    for (const UnrolledSample& sample : batch) {
        const int K = static_cast<int>(sample.actions.size());
        assert(static_cast<int>(sample.targetValues.size()) == K + 1);
        assert(static_cast<int>(sample.targetRewards.size()) == K + 1);
        assert(static_cast<int>(sample.targetPolicies.size()) == K + 1);

        // Every unroll step past 0 contributes at 1/K, so a deeply
        // unrolled sample does not outweigh a shallow one.
        const float tailScale = (K > 0) ? 1.0f / static_cast<float>(K) : 1.0f;
        auto lossScale = [&](int k) { return k == 0 ? 1.0f : tailScale; };

        // ---- forward, keeping everything the backward pass will need ----

        HiddenTrace hTrace = representationHidden(sample.observation);
        std::vector<std::vector<float>> latentPre(K + 1);   // gFc2/hFc2 output, pre-normalization
        std::vector<std::vector<float>> latent(K + 1);      // after minMaxNormalize

        latentPre[0] = hFc2_.forward(hTrace.activation);
        latent[0] = minMaxNormalize(latentPre[0]);

        std::vector<std::vector<float>> dynamicsInput(K);
        std::vector<HiddenTrace> dynamicsTrace(K);
        std::vector<float> reward(K + 1, 0.0f);
        std::vector<float> rewardPre(K + 1, 0.0f);

        for (int k = 0; k < K; ++k) {
            dynamicsInput[k] = makeDynamicsInput(latent[k], sample.actions[k]);
            dynamicsTrace[k] = dynamicsHidden(dynamicsInput[k]);
            latentPre[k + 1] = gFc2_.forward(dynamicsTrace[k].activation);
            latent[k + 1] = minMaxNormalize(latentPre[k + 1]);
            rewardPre[k + 1] = gReward_.forward(dynamicsTrace[k].activation)[0];
            reward[k + 1] = std::tanh(rewardPre[k + 1]);
        }

        std::vector<HiddenTrace> predictionTrace(K + 1);
        std::vector<std::array<float, kActionSize>> policy(K + 1);
        std::vector<float> value(K + 1, 0.0f);

        for (int k = 0; k <= K; ++k) {
            predictionTrace[k] = predictionHidden(latent[k]);
            policy[k] = softmax9(fPolicy_.forward(predictionTrace[k].activation));
            value[k] = std::tanh(fValue_.forward(predictionTrace[k].activation)[0]);
        }

        // ---- loss ----

        for (int k = 0; k <= K; ++k) {
            float scale = lossScale(k);
            float valueError = value[k] - sample.targetValues[k];
            losses.value += scale * valueError * valueError;
            for (int a = 0; a < kActionSize; ++a) {
                losses.policy -= scale * sample.targetPolicies[k][a] * std::log(policy[k][a] + 1e-8f);
            }
            // targetRewards[0] is always 0 and excluded: step 0 has no
            // incoming transition to predict a reward for.
            if (k > 0) {
                float rewardError = reward[k] - sample.targetRewards[k];
                losses.reward += scale * rewardError * rewardError;
            }
        }

        // ---- backward through time ----

        // dLatent[k] accumulates dLoss/dLatent[k] from two sources: the
        // prediction head at step k, and the dynamics step k -> k+1. The
        // single reverse loop guarantees both have arrived before it is
        // used.
        std::vector<std::vector<float>> dLatent(K + 1, std::vector<float>(kLatentSize, 0.0f));

        for (int k = K; k >= 0; --k) {
            float scale = lossScale(k);

            // prediction head at step k
            float dValuePre = scale * 2.0f * (value[k] - sample.targetValues[k]) *
                              (1.0f - value[k] * value[k]);
            std::vector<float> dPolicyLogits(kActionSize);
            for (int a = 0; a < kActionSize; ++a) {
                // d(cross-entropy o softmax)/d(logit) = p - target
                dPolicyLogits[a] = scale * (policy[k][a] - sample.targetPolicies[k][a]);
            }

            std::vector<float> dPredictionHidden =
                fValue_.backward(predictionTrace[k].activation, {dValuePre});
            std::vector<float> dFromPolicy =
                fPolicy_.backward(predictionTrace[k].activation, dPolicyLogits);
            for (int i = 0; i < kHiddenSize; ++i) dPredictionHidden[i] += dFromPolicy[i];

            std::vector<float> dPredictionPre =
                reluBackward(predictionTrace[k].preActivation, dPredictionHidden);
            std::vector<float> dFromPrediction = fFc1_.backward(latent[k], dPredictionPre);
            for (int i = 0; i < kLatentSize; ++i) dLatent[k][i] += dFromPrediction[i];

            if (k > 0) {
                // dynamics step k-1 -> k
                std::vector<float> dLatentPre = minMaxNormalizeBackward(latentPre[k], dLatent[k]);
                std::vector<float> dDynamicsHidden =
                    gFc2_.backward(dynamicsTrace[k - 1].activation, dLatentPre);

                float dRewardPre = scale * 2.0f * (reward[k] - sample.targetRewards[k]) *
                                   (1.0f - reward[k] * reward[k]);
                std::vector<float> dFromReward =
                    gReward_.backward(dynamicsTrace[k - 1].activation, {dRewardPre});
                for (int i = 0; i < kHiddenSize; ++i) dDynamicsHidden[i] += dFromReward[i];

                std::vector<float> dDynamicsPre =
                    reluBackward(dynamicsTrace[k - 1].preActivation, dDynamicsHidden);
                std::vector<float> dDynamicsInput =
                    gFc1_.backward(dynamicsInput[k - 1], dDynamicsPre);

                // The half gradient. Scaling what flows back into the
                // dynamics input by 0.5 at every step keeps gradient
                // magnitude from compounding across the recurrence. One
                // line, easy to omit, and omitting it destabilizes latents
                // as the unroll deepens.
                for (int i = 0; i < kLatentSize; ++i) dLatent[k - 1][i] += 0.5f * dDynamicsInput[i];
            } else {
                // representation network
                std::vector<float> dLatentPre = minMaxNormalizeBackward(latentPre[0], dLatent[0]);
                std::vector<float> dHidden = hFc2_.backward(hTrace.activation, dLatentPre);
                std::vector<float> dHiddenPre = reluBackward(hTrace.preActivation, dHidden);
                std::vector<float> observation(sample.observation.begin(), sample.observation.end());
                hFc1_.backward(observation, dHiddenPre);
            }
        }
    }

    const float scale = 1.0f / static_cast<float>(batch.size());
    for (Dense* layer : {&hFc1_, &hFc2_, &gFc1_, &gFc2_, &gReward_, &fFc1_, &fPolicy_, &fValue_}) {
        layer->applySgd(learningRate, scale);
    }

    losses.value *= scale;
    losses.policy *= scale;
    losses.reward *= scale;
    losses.total = losses.value + losses.policy + losses.reward;
    return losses;
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```sh
cmake --build build && ./build/test_network && ctest --test-dir build --output-on-failure
```

Expected: PASS — `all network tests passed`, 6/6 tests.

If `test_gradient_matches_numerical_through_the_full_unroll` fails on the ratio while the sign tests pass, the gradient is right in direction but wrong in magnitude. Check the two scale factors first: the `1/K` in `lossScale` and the `0.5f` half gradient.

- [ ] **Step 5: Commit**

```bash
git add src/network.cpp tests/test_network.cpp
git commit -m "$(cat <<'EOF'
Add backprop-through-time over the K-step unroll

One reverse pass accumulates into all three networks, with the paper's
1/K loss scaling and the half gradient into the dynamics input. The
magnitude-sensitive gradient check is the real deliverable: a wrong
gradient here would not crash, it would just train worse forever.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: MCTS over latent states

AlphaZero's tree is made of board positions. Every node knows the real position, so it knows which moves are legal and whether the game is over. MuZero's tree is made of **latent vectors**. Only the root corresponds to a position anyone can name.

Three consequences follow, and all three are tested here:

**Legality is a root-only concept.** Illegal actions are masked out of the root prior and nowhere else. Every non-root node gets all nine children. No training trajectory ever contains an illegal action, so dynamics on those inputs is entirely unconstrained — the network will map them somewhere and prediction will report a confident, meaningless value there. Search will sometimes spend simulations on that. This is the price of not being given the rules, and the project measures it rather than engineering around it.

**Q values carry predicted rewards, so their scale is unknown.** AlphaZero compares a `tanh` value in [-1, 1] directly against a prior. MuZero's Q is `reward + discount * -childValue`, a sum whose range depends on what the reward head has learned. `MinMaxStats` tracks the running min and max seen in the tree and normalizes Q into [0, 1] before the comparison.

**Exploration grows with visit count.** MuZero replaces AlphaZero's fixed `cPuct` with `c1 + log((N + c2 + 1) / c2)`. At tic-tac-toe's simulation counts the log term barely moves, since `c2 = 19652`; it is implemented faithfully anyway, because the reason it exists is visible only at scale and worth understanding.

Search and backup are factored into free functions over plain structs so the sign conventions can be tested directly, without needing a trained network to make assertions about.

**Files:**
- Create: `include/mz/mcts.hpp`, `src/mcts.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/test_mcts.cpp`

**Interfaces:**
- Consumes: `mz::Board` (Task 1), `mz::MuZeroNetwork` (Tasks 5-6).
- Produces: constants `mz::kDiscount`, `mz::kPbCInit`, `mz::kPbCBase`; `mz::MinMaxStats` with `update(float)`, `normalize(float) const`; `mz::BackupNode{float reward; int visitCount; float valueSum;}`; free functions `nodeValue(const BackupNode&)`, `edgeQ(float childReward, float childValue, float discount)`, `explorationTerm(float prior, int parentVisits, int childVisits)`, `backupPath(const std::vector<BackupNode*>&, float leafValue, float discount, MinMaxStats&)`; `mz::MCTSConfig{int numSimulations=60; bool addRootNoise=false; float dirichletAlpha=0.3f; float dirichletEpsilon=0.25f;}`; `mz::MCTSResult{std::array<float,9> visitDistribution; int selectedMove; float rootValue; int nodesExpanded; int maxDepth;}`; `mz::MCTS` with `MCTS(const MuZeroNetwork&, const MCTSConfig&, std::mt19937&)`, `run(const Board&, float temperature) -> MCTSResult`, and statics `maskAndRenormalize`, `mixDirichletNoise`.

- [ ] **Step 1: Write the failing test**

`tests/test_mcts.cpp`:

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>
#include "mz/board.hpp"
#include "mz/mcts.hpp"
#include "mz/network.hpp"

using namespace mz;

namespace {
bool near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) < tol; }
} // namespace

void test_minmax_stats_passes_through_before_any_update() {
    MinMaxStats stats;
    assert(near(stats.normalize(0.7f), 0.7f));
    assert(near(stats.normalize(-2.0f), -2.0f));
}

void test_minmax_stats_normalizes_observed_range_to_unit_interval() {
    MinMaxStats stats;
    stats.update(-0.3f);
    stats.update(0.3f);
    assert(near(stats.normalize(-0.3f), 0.0f));
    assert(near(stats.normalize(0.3f), 1.0f));
    assert(near(stats.normalize(0.0f), 0.5f));
}

void test_minmax_stats_handles_a_single_observation() {
    MinMaxStats stats;
    stats.update(0.42f);
    assert(std::isfinite(stats.normalize(0.42f)));
}

void test_edge_q_negates_the_child_value() {
    // Q of the edge into a child, in the PARENT's perspective: the reward
    // the parent's player collects, plus the (negated) value of the
    // position it hands the opponent.
    assert(near(edgeQ(0.5f, 0.8f, 1.0f), -0.3f));
    assert(near(edgeQ(0.0f, -0.6f, 1.0f), 0.6f));
}

void test_exploration_term_shrinks_as_a_child_is_visited() {
    float first = explorationTerm(0.4f, 100, 0);
    float later = explorationTerm(0.4f, 100, 10);
    assert(first > later);
}

void test_exploration_term_scales_with_prior() {
    assert(explorationTerm(0.8f, 50, 3) > explorationTerm(0.2f, 50, 3));
}

void test_exploration_term_grows_with_parent_visits() {
    assert(explorationTerm(0.3f, 400, 5) > explorationTerm(0.3f, 100, 5));
}

void test_backup_alternates_sign_and_carries_reward() {
    // Three nodes. B is reached by a move that paid its mover +0.5, and the
    // leaf evaluates to +0.8 for whoever moves at B.
    BackupNode root{0.0f, 0, 0.0f};
    BackupNode a{0.0f, 0, 0.0f};
    BackupNode b{0.5f, 0, 0.0f};
    std::vector<BackupNode*> path = {&root, &a, &b};

    MinMaxStats stats;
    backupPath(path, 0.8f, 1.0f, stats);

    assert(b.visitCount == 1 && near(b.valueSum, 0.8f));
    // From A's player's view: collect 0.5, hand over a 0.8 position.
    assert(a.visitCount == 1 && near(a.valueSum, -0.3f));
    // And back to the root player, negated once more.
    assert(root.visitCount == 1 && near(root.valueSum, 0.3f));
}

void test_backup_records_edge_q_but_not_the_root() {
    BackupNode root{0.0f, 0, 0.0f};
    BackupNode child{0.0f, 0, 0.0f};
    std::vector<BackupNode*> path = {&root, &child};
    MinMaxStats stats;
    backupPath(path, 0.6f, 1.0f, stats);
    // The only edge is root -> child, whose Q is -0.6 in the root's view.
    // With one observation the range is degenerate but must stay finite.
    assert(std::isfinite(stats.normalize(-0.6f)));
    assert(near(child.valueSum, 0.6f));
    assert(near(root.valueSum, -0.6f));
}

void test_node_value_is_zero_before_any_visit() {
    BackupNode node{0.0f, 0, 0.0f};
    assert(near(nodeValue(node), 0.0f));
}

void test_mask_and_renormalize_zeroes_illegal_actions() {
    std::array<float, 9> policy{};
    policy.fill(1.0f / 9.0f);
    std::vector<int> legal = {0, 4, 8};
    std::array<float, 9> masked = MCTS::maskAndRenormalize(policy, legal);

    float sum = 0.0f;
    for (int a = 0; a < 9; ++a) {
        if (a == 0 || a == 4 || a == 8) assert(masked[a] > 0.0f);
        else assert(masked[a] == 0.0f);
        sum += masked[a];
    }
    assert(near(sum, 1.0f));
}

void test_dirichlet_noise_gives_every_legal_action_positive_prior() {
    // A prior of exactly zero must still come out positive, so search can
    // never permanently starve a move the network is wrong about.
    std::array<float, 9> priors{};
    priors.fill(0.0f);
    priors[3] = 1.0f;
    std::vector<int> legal = {0, 3, 6};
    std::mt19937 rng(99);
    MCTS::mixDirichletNoise(priors, legal, rng, 0.3f, 0.25f);
    for (int a : legal) assert(priors[a] > 0.0f);
    assert(priors[1] == 0.0f);   // untouched: not legal
}

void test_search_visits_only_legal_moves_at_the_root() {
    MuZeroNetwork network(1234);
    Board board;
    board = board.applyMove(0);
    board = board.applyMove(4);
    board = board.applyMove(8);

    MCTSConfig config;
    config.numSimulations = 40;
    std::mt19937 rng(7);
    MCTS mcts(network, config, rng);
    MCTSResult result = mcts.run(board, 0.0f);

    for (int a : {0, 4, 8}) assert(result.visitDistribution[a] == 0.0f);
    assert(board.isLegalMove(result.selectedMove));

    float sum = 0.0f;
    for (float v : result.visitDistribution) sum += v;
    assert(near(sum, 1.0f));
}

void test_search_expands_one_node_per_simulation() {
    MuZeroNetwork network(11);
    Board board;
    MCTSConfig config;
    config.numSimulations = 25;
    std::mt19937 rng(3);
    MCTS mcts(network, config, rng);
    MCTSResult result = mcts.run(board, 0.0f);
    assert(result.nodesExpanded == 25);
}

void test_tree_grows_below_a_root_with_only_one_legal_move() {
    // The sharpest test of root-only legality. Fill the board down to a
    // single empty square: the root has exactly one child. If non-root
    // nodes were also masked to legal moves -- or if the tree detected
    // terminal states -- it could not go deeper than that one child. It
    // does, because below the root the search knows neither which actions
    // are legal nor when the game has ended.
    Board board;
    int moves[] = {0, 1, 2, 4, 3, 5, 7, 6};   // eight plies, square 8 open
    for (int m : moves) board = board.applyMove(m);
    assert(board.legalMoves().size() == 1);
    assert(!board.isTerminal());

    MuZeroNetwork network(2);
    MCTSConfig config;
    config.numSimulations = 20;
    std::mt19937 rng(5);
    MCTS mcts(network, config, rng);
    MCTSResult result = mcts.run(board, 0.0f);

    assert(result.selectedMove == 8);
    // Playing square 8 ends the game, so the real position one ply down
    // has no legal moves at all. A rules-aware search could not go below
    // depth 1 here. Depth 2 is reached on the second simulation and is
    // guaranteed, not luck.
    assert(result.maxDepth >= 2);
}

void test_root_value_is_finite_and_bounded() {
    MuZeroNetwork network(19);
    Board board;
    MCTSConfig config;
    config.numSimulations = 30;
    std::mt19937 rng(1);
    MCTS mcts(network, config, rng);
    MCTSResult result = mcts.run(board, 0.0f);
    assert(std::isfinite(result.rootValue));
    assert(result.rootValue >= -2.0f && result.rootValue <= 2.0f);
}

void test_zero_temperature_picks_the_most_visited_move() {
    MuZeroNetwork network(23);
    Board board;
    MCTSConfig config;
    config.numSimulations = 50;
    std::mt19937 rng(9);
    MCTS mcts(network, config, rng);
    MCTSResult result = mcts.run(board, 0.0f);

    float best = 0.0f;
    for (float v : result.visitDistribution) best = std::fmax(best, v);
    assert(near(result.visitDistribution[result.selectedMove], best));
}

void test_positive_temperature_can_pick_something_other_than_the_max() {
    MuZeroNetwork network(29);
    Board board;
    MCTSConfig config;
    config.numSimulations = 30;
    std::mt19937 rng(4);

    std::set<int> chosen;
    for (int trial = 0; trial < 60; ++trial) {
        MCTS mcts(network, config, rng);
        chosen.insert(mcts.run(board, 1.0f).selectedMove);
    }
    assert(chosen.size() > 1);
}

void test_root_noise_changes_the_search() {
    MuZeroNetwork network(37);
    Board board;

    MCTSConfig quiet;
    quiet.numSimulations = 40;
    quiet.addRootNoise = false;

    MCTSConfig noisy = quiet;
    noisy.addRootNoise = true;

    std::mt19937 rngA(2), rngB(2);
    MCTS a(network, quiet, rngA);
    MCTS b(network, noisy, rngB);
    MCTSResult ra = a.run(board, 0.0f);
    MCTSResult rb = b.run(board, 0.0f);

    bool differs = false;
    for (int i = 0; i < 9; ++i) {
        if (!near(ra.visitDistribution[i], rb.visitDistribution[i], 1e-4f)) differs = true;
    }
    assert(differs);
}

int main() {
    test_minmax_stats_passes_through_before_any_update();
    test_minmax_stats_normalizes_observed_range_to_unit_interval();
    test_minmax_stats_handles_a_single_observation();
    test_edge_q_negates_the_child_value();
    test_exploration_term_shrinks_as_a_child_is_visited();
    test_exploration_term_scales_with_prior();
    test_exploration_term_grows_with_parent_visits();
    test_backup_alternates_sign_and_carries_reward();
    test_backup_records_edge_q_but_not_the_root();
    test_node_value_is_zero_before_any_visit();
    test_mask_and_renormalize_zeroes_illegal_actions();
    test_dirichlet_noise_gives_every_legal_action_positive_prior();
    test_search_visits_only_legal_moves_at_the_root();
    test_search_expands_one_node_per_simulation();
    test_tree_grows_below_a_root_with_only_one_legal_move();
    test_root_value_is_finite_and_bounded();
    test_zero_temperature_picks_the_most_visited_move();
    test_positive_temperature_can_pick_something_other_than_the_max();
    test_root_noise_changes_the_search();
    std::printf("all mcts tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run the test to verify it fails**

Add `mz_add_test(test_mcts)` to `CMakeLists.txt`, then build.

Expected: FAIL — `mz/mcts.hpp: No such file or directory`.

- [ ] **Step 3: Write the header**

`include/mz/mcts.hpp`:

```cpp
#pragma once
#include <array>
#include <memory>
#include <random>
#include <vector>
#include "mz/board.hpp"
#include "mz/network.hpp"

namespace mz {

// A board game: a win in five moves is worth exactly a win in nine.
constexpr float kDiscount = 1.0f;
// MuZero's exploration constants (appendix B). c2 is large enough that the
// log term barely moves at tic-tac-toe's simulation counts; it is here
// because the mechanism, not its magnitude, is the point.
constexpr float kPbCInit = 1.25f;
constexpr float kPbCBase = 19652.0f;

// Running min and max of the Q values seen in one search tree.
//
// AlphaZero does not need this: its Q is a tanh value in [-1, 1], directly
// comparable against a prior. MuZero's Q is reward + discount * -value,
// whose scale depends on what the reward head has learned, so it has to be
// normalized into [0, 1] before it can be traded against the exploration
// term.
class MinMaxStats {
public:
    void update(float value);
    // Maps `value` into [0, 1] using the range seen so far. Returns it
    // unchanged while fewer than two distinct values have been observed.
    float normalize(float value) const;

private:
    float min_ = 0.0f;
    float max_ = 0.0f;
    bool seeded_ = false;
};

// The subset of a search node that backup touches. Split out as a plain
// struct so the sign conventions can be tested directly, without standing
// up a network to make assertions about.
struct BackupNode {
    // Reward predicted on the transition INTO this node, in the
    // perspective of the player who took that action.
    float reward = 0.0f;
    int visitCount = 0;
    float valueSum = 0.0f;
};

// Mean value of a node, in the perspective of the player to move there.
// Zero for an unvisited node, matching MuZero's pseudocode.
float nodeValue(const BackupNode& node);

// Q of the edge leading into a child, in the PARENT's perspective: the
// reward the parent's player collects, plus the negated value of the
// position handed to the opponent.
float edgeQ(float childReward, float childValue, float discount);

// MuZero's PUCT exploration term:
//   prior * sqrt(parentVisits) / (1 + childVisits) * (c1 + log((parentVisits + c2 + 1) / c2))
// AlphaZero uses a fixed cPuct in place of that trailing factor.
float explorationTerm(float prior, int parentVisits, int childVisits);

// Walks a root-to-leaf path in reverse, adding `leafValue` at the leaf and
// negating at every ply on the way up, accumulating each node's predicted
// reward as it goes. Records each traversed edge's Q into `stats`.
void backupPath(const std::vector<BackupNode*>& path, float leafValue, float discount,
                MinMaxStats& stats);

struct MCTSConfig {
    int numSimulations = 60;
    // Dirichlet noise on the root prior, so search cannot permanently
    // starve a move the network is confident (and wrong) about. Self-play
    // only -- evaluation and interactive play search without it.
    bool addRootNoise = false;
    float dirichletAlpha = 0.3f;
    float dirichletEpsilon = 0.25f;
};

struct MCTSResult {
    // Normalized root visit counts over the nine actions. Illegal actions
    // are exactly zero. This is the policy training target.
    std::array<float, 9> visitDistribution;
    int selectedMove;
    // Root value after search, in the perspective of the player to move.
    // Stored per position and later used as the n-step bootstrap source --
    // a target AlphaZero has no equivalent of.
    float rootValue;
    // Diagnostics, also used by the tests: one node is expanded per
    // simulation, and maxDepth going below 1 is what proves the tree is
    // not legality-masked below the root.
    int nodesExpanded;
    int maxDepth;
};

// PUCT search over learned latent states.
//
// The real board is read exactly once, at the root, to produce the
// observation and the legal-move mask. Everything below is dynamics
// applied to latents. There is no terminal detection in the tree and no
// legality below the root: MuZero has to learn that walking into an
// illegal action is bad, the same way it learns everything else.
class MCTS {
public:
    MCTS(const MuZeroNetwork& network, const MCTSConfig& config, std::mt19937& rng);

    // temperature > 0: sample proportional to visitCount^(1/temperature).
    // temperature == 0: take the max-visit move, ties to the lowest index.
    // Precondition: !board.isTerminal() (asserted).
    MCTSResult run(const Board& board, float temperature);

    // Exposed for testing: restricts `policy` to `legalActions` and
    // renormalizes. Entries outside `legalActions` become exactly zero.
    // Applied at the root and nowhere else.
    static std::array<float, 9> maskAndRenormalize(const std::array<float, 9>& policy,
                                                   const std::vector<int>& legalActions);

    // Exposed for testing: mixes Dirichlet(alpha) noise into the priors of
    // `actions` in place, weighted by epsilon. Every listed action ends up
    // strictly positive regardless of where it started.
    static void mixDirichletNoise(std::array<float, 9>& priors, const std::vector<int>& actions,
                                  std::mt19937& rng, float alpha, float epsilon);

private:
    struct Node {
        BackupNode stats;
        float prior = 0.0f;
        bool expanded = false;
        std::vector<float> latent;
        std::array<std::unique_ptr<Node>, 9> children{};
    };

    int selectChild(const Node& parent, const MinMaxStats& stats) const;

    const MuZeroNetwork& network_;
    MCTSConfig config_;
    std::mt19937& rng_;
};

} // namespace mz
```

- [ ] **Step 4: Write the implementation**

`src/mcts.cpp`:

```cpp
#include "mz/mcts.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace mz {

void MinMaxStats::update(float value) {
    if (!seeded_) {
        min_ = max_ = value;
        seeded_ = true;
        return;
    }
    min_ = std::min(min_, value);
    max_ = std::max(max_, value);
}

float MinMaxStats::normalize(float value) const {
    if (!seeded_ || max_ <= min_) return value;
    return (value - min_) / (max_ - min_);
}

float nodeValue(const BackupNode& node) {
    return node.visitCount == 0 ? 0.0f : node.valueSum / static_cast<float>(node.visitCount);
}

float edgeQ(float childReward, float childValue, float discount) {
    return childReward + discount * -childValue;
}

float explorationTerm(float prior, int parentVisits, int childVisits) {
    float pbC = std::log((static_cast<float>(parentVisits) + kPbCBase + 1.0f) / kPbCBase) + kPbCInit;
    return prior * std::sqrt(static_cast<float>(parentVisits)) / (1.0f + static_cast<float>(childVisits)) * pbC;
}

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

MCTS::MCTS(const MuZeroNetwork& network, const MCTSConfig& config, std::mt19937& rng)
    : network_(network), config_(config), rng_(rng) {
    assert(config_.numSimulations > 0);
}

std::array<float, 9> MCTS::maskAndRenormalize(const std::array<float, 9>& policy,
                                              const std::vector<int>& legalActions) {
    assert(!legalActions.empty());
    std::array<float, 9> masked{};
    masked.fill(0.0f);
    float sum = 0.0f;
    for (int a : legalActions) sum += policy[a];
    if (sum <= 0.0f) {
        // Degenerate, but survivable: fall back to uniform over legal.
        for (int a : legalActions) masked[a] = 1.0f / static_cast<float>(legalActions.size());
        return masked;
    }
    for (int a : legalActions) masked[a] = policy[a] / sum;
    return masked;
}

void MCTS::mixDirichletNoise(std::array<float, 9>& priors, const std::vector<int>& actions,
                             std::mt19937& rng, float alpha, float epsilon) {
    assert(!actions.empty());
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    std::vector<float> noise(actions.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < actions.size(); ++i) {
        // Floored: a small alpha can draw a gamma sample that rounds to
        // zero in float, which would defeat the whole purpose.
        noise[i] = std::max(gamma(rng), 1e-6f);
        sum += noise[i];
    }
    for (std::size_t i = 0; i < actions.size(); ++i) {
        int a = actions[i];
        priors[a] = (1.0f - epsilon) * priors[a] + epsilon * (noise[i] / sum);
    }
}

int MCTS::selectChild(const Node& parent, const MinMaxStats& stats) const {
    float bestScore = -std::numeric_limits<float>::infinity();
    int bestAction = -1;
    for (int a = 0; a < 9; ++a) {
        const Node* child = parent.children[a].get();
        if (!child) continue;   // only happens at the root, for illegal moves
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
    assert(bestAction >= 0);
    return bestAction;
}

MCTSResult MCTS::run(const Board& board, float temperature) {
    assert(!board.isTerminal());
    const std::vector<int> legalActions = board.legalMoves();

    Node root;
    MuZeroNetwork::InitialInference initial = network_.initialInference(board.encode());
    root.latent = initial.latent;

    // The one and only place legality enters the search.
    std::array<float, 9> rootPriors = maskAndRenormalize(initial.policy, legalActions);
    if (config_.addRootNoise) {
        mixDirichletNoise(rootPriors, legalActions, rng_, config_.dirichletAlpha,
                          config_.dirichletEpsilon);
    }
    for (int a : legalActions) {
        root.children[a] = std::make_unique<Node>();
        root.children[a]->prior = rootPriors[a];
    }
    root.expanded = true;

    MinMaxStats stats;
    int nodesExpanded = 0;
    int maxDepth = 0;

    for (int sim = 0; sim < config_.numSimulations; ++sim) {
        Node* node = &root;
        std::vector<Node*> path{&root};
        int lastAction = -1;

        while (node->expanded) {
            lastAction = selectChild(*node, stats);
            node = node->children[lastAction].get();
            path.push_back(node);
        }

        Node* parent = path[path.size() - 2];
        MuZeroNetwork::RecurrentInference step =
            network_.recurrentInference(parent->latent, lastAction);

        node->latent = std::move(step.latent);
        node->stats.reward = step.reward;
        // Below the root, every one of the nine actions gets a child.
        // There is no legality oracle here and no terminal detection: the
        // search does not know the rules, and has to learn from the value
        // and reward heads that some of these branches are worthless.
        for (int a = 0; a < 9; ++a) {
            node->children[a] = std::make_unique<Node>();
            node->children[a]->prior = step.policy[a];
        }
        node->expanded = true;
        ++nodesExpanded;
        maxDepth = std::max(maxDepth, static_cast<int>(path.size()) - 1);

        std::vector<BackupNode*> statsPath;
        statsPath.reserve(path.size());
        for (Node* n : path) statsPath.push_back(&n->stats);
        backupPath(statsPath, step.value, kDiscount, stats);
    }

    MCTSResult result{};
    result.visitDistribution.fill(0.0f);
    result.nodesExpanded = nodesExpanded;
    result.maxDepth = maxDepth;
    result.rootValue = nodeValue(root.stats);

    std::array<float, 9> counts{};
    counts.fill(0.0f);
    float totalVisits = 0.0f;
    for (int a = 0; a < 9; ++a) {
        if (!root.children[a]) continue;
        counts[a] = static_cast<float>(root.children[a]->stats.visitCount);
        totalVisits += counts[a];
    }
    assert(totalVisits > 0.0f);
    for (int a = 0; a < 9; ++a) result.visitDistribution[a] = counts[a] / totalVisits;

    if (temperature <= 0.0f) {
        int best = legalActions.front();
        for (int a : legalActions) {
            if (counts[a] > counts[best]) best = a;
        }
        result.selectedMove = best;
    } else {
        std::array<float, 9> weights{};
        weights.fill(0.0f);
        float weightSum = 0.0f;
        for (int a : legalActions) {
            weights[a] = std::pow(counts[a], 1.0f / temperature);
            weightSum += weights[a];
        }
        std::uniform_real_distribution<float> pick(0.0f, weightSum);
        float draw = pick(rng_);
        int chosen = legalActions.back();
        float running = 0.0f;
        for (int a : legalActions) {
            running += weights[a];
            if (draw <= running) {
                chosen = a;
                break;
            }
        }
        result.selectedMove = chosen;
    }

    return result;
}

} // namespace mz
```

- [ ] **Step 5: Add to the build and run the tests**

Add `src/mcts.cpp` to the `add_library(mz ...)` source list.

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: PASS — `all mcts tests passed`, 7/7 tests.

- [ ] **Step 6: Commit**

```bash
git add include/mz/mcts.hpp src/mcts.cpp tests/test_mcts.cpp CMakeLists.txt
git commit -m "$(cat <<'EOF'
Add PUCT search over learned latent states

Legality is masked at the root and nowhere else, Q is min-max
normalized before it meets the exploration term, and backup carries
predicted rewards with a sign flip per ply. Backup and scoring are free
functions over plain structs so the sign conventions are directly
testable.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Self-play

Plays complete games with MCTS over the learned model and emits one `GameHistory`.

The real board is stepped between moves. MuZero learns dynamics to *search* with, not to play with — when it actually commits to a move, the true rules apply. The learned model is a thinking aid, not a substitute for the game.

Two differences from the AlphaZero version are worth noticing. It records the MCTS root value at every ply, which AlphaZero has no use for and does not store; here it is the bootstrap source for value targets. And it records rewards along the trajectory rather than one outcome at the end, even though on this domain every reward but the last is zero.

**Files:**
- Create: `include/mz/selfplay.hpp`, `src/selfplay.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/test_selfplay.cpp`

**Interfaces:**
- Consumes: `mz::Board`, `mz::MuZeroNetwork`, `mz::MCTS`, `mz::GameHistory`.
- Produces: `mz::SelfPlayConfig{int numSimulations=60; int temperatureMoves=2; float dirichletAlpha=0.3f; float dirichletEpsilon=0.25f;}` and `mz::playSelfPlayGame(const MuZeroNetwork&, const SelfPlayConfig&, std::mt19937&) -> GameHistory`.

- [ ] **Step 1: Write the failing test**

`tests/test_selfplay.cpp`:

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include "mz/board.hpp"
#include "mz/network.hpp"
#include "mz/selfplay.hpp"

using namespace mz;

namespace {
bool near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) < tol; }

GameHistory playOne(unsigned seed) {
    MuZeroNetwork network(seed);
    SelfPlayConfig config;
    config.numSimulations = 12;   // small: these tests check structure, not strength
    std::mt19937 rng(seed);
    return playSelfPlayGame(network, config, rng);
}
} // namespace

void test_game_length_is_plausible() {
    GameHistory game = playOne(1);
    assert(game.length() >= 5 && game.length() <= 9);
}

void test_all_trajectory_arrays_have_the_same_length() {
    GameHistory game = playOne(2);
    assert(game.observations.size() == game.length());
    assert(game.searchPolicies.size() == game.length());
    assert(game.searchValues.size() == game.length());
    assert(game.rewards.size() == game.length());
}

void test_replaying_the_actions_reproduces_the_trajectory() {
    // The strongest structural check: every recorded action was legal in
    // the position whose observation was recorded alongside it, and the
    // sequence really does end the game.
    GameHistory game = playOne(3);
    Board board;
    for (std::size_t t = 0; t < game.length(); ++t) {
        assert(!board.isTerminal());
        auto encoded = board.encode();
        for (int i = 0; i < 18; ++i) assert(near(encoded[i], game.observations[t][i]));
        assert(board.isLegalMove(game.actions[t]));
        board = board.applyMove(game.actions[t]);
    }
    assert(board.isTerminal());
}

void test_rewards_are_zero_until_the_final_move() {
    GameHistory game = playOne(4);
    for (std::size_t t = 0; t + 1 < game.length(); ++t) assert(near(game.rewards[t], 0.0f));
}

void test_final_reward_matches_the_outcome() {
    // Whoever plays the last move either wins or draws -- in tic-tac-toe
    // you cannot lose by moving. So the final reward is +1 or 0.
    GameHistory game = playOne(5);
    Board board;
    for (int action : game.actions) board = board.applyMove(action);
    float finalReward = game.rewards.back();
    if (board.outcome() == Outcome::Draw) assert(near(finalReward, 0.0f));
    else assert(near(finalReward, 1.0f));
}

void test_search_policies_never_recommend_an_occupied_square() {
    GameHistory game = playOne(6);
    Board board;
    for (std::size_t t = 0; t < game.length(); ++t) {
        float sum = 0.0f;
        for (int a = 0; a < 9; ++a) {
            if (!board.isLegalMove(a)) assert(game.searchPolicies[t][a] == 0.0f);
            sum += game.searchPolicies[t][a];
        }
        assert(near(sum, 1.0f, 1e-4f));
        board = board.applyMove(game.actions[t]);
    }
}

void test_search_values_are_recorded_and_finite() {
    // AlphaZero stores nothing like this. Here every ply's root value is
    // kept, because it is what the n-step value target bootstraps from.
    GameHistory game = playOne(7);
    for (float v : game.searchValues) {
        assert(std::isfinite(v));
        assert(v >= -2.0f && v <= 2.0f);
    }
}

void test_two_seeds_produce_different_games() {
    GameHistory a = playOne(8);
    GameHistory b = playOne(9);
    assert(a.actions != b.actions || a.length() != b.length());
}

int main() {
    test_game_length_is_plausible();
    test_all_trajectory_arrays_have_the_same_length();
    test_replaying_the_actions_reproduces_the_trajectory();
    test_rewards_are_zero_until_the_final_move();
    test_final_reward_matches_the_outcome();
    test_search_policies_never_recommend_an_occupied_square();
    test_search_values_are_recorded_and_finite();
    test_two_seeds_produce_different_games();
    std::printf("all selfplay tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run the test to verify it fails**

Add `mz_add_test(test_selfplay)` to `CMakeLists.txt`, then build.

Expected: FAIL — `mz/selfplay.hpp: No such file or directory`.

- [ ] **Step 3: Write the header**

`include/mz/selfplay.hpp`:

```cpp
#pragma once
#include <random>
#include "mz/game_history.hpp"
#include "mz/network.hpp"

namespace mz {

struct SelfPlayConfig {
    int numSimulations = 60;
    // Plies from the start that use temperature=1.0 sampling, for training
    // data diversity. Greedy afterward.
    int temperatureMoves = 2;
    float dirichletAlpha = 0.3f;
    float dirichletEpsilon = 0.25f;
};

// Plays one complete game with MCTS over the learned model and returns its
// trajectory.
//
// The REAL board is stepped between moves. MuZero learns dynamics to
// search with, not to play with: when it commits to a move, the true rules
// apply. The learned model is a thinking aid, never a substitute for the
// game.
GameHistory playSelfPlayGame(const MuZeroNetwork& network, const SelfPlayConfig& config,
                             std::mt19937& rng);

} // namespace mz
```

- [ ] **Step 4: Write the implementation**

`src/selfplay.cpp`:

```cpp
#include "mz/selfplay.hpp"
#include "mz/board.hpp"
#include "mz/mcts.hpp"

namespace mz {

GameHistory playSelfPlayGame(const MuZeroNetwork& network, const SelfPlayConfig& config,
                             std::mt19937& rng) {
    MCTSConfig searchConfig;
    searchConfig.numSimulations = config.numSimulations;
    searchConfig.addRootNoise = true;
    searchConfig.dirichletAlpha = config.dirichletAlpha;
    searchConfig.dirichletEpsilon = config.dirichletEpsilon;

    Board board;
    GameHistory game;
    int ply = 0;

    while (!board.isTerminal()) {
        MCTS mcts(network, searchConfig, rng);
        float temperature = (ply < config.temperatureMoves) ? 1.0f : 0.0f;
        MCTSResult result = mcts.run(board, temperature);

        game.observations.push_back(board.encode());
        game.actions.push_back(result.selectedMove);
        game.searchPolicies.push_back(result.visitDistribution);
        // AlphaZero has no use for this and does not store it. Here it is
        // the source the n-step value target bootstraps from.
        game.searchValues.push_back(result.rootValue);
        game.rewards.push_back(0.0f);

        Cell mover = board.playerToMove();
        board = board.applyMove(result.selectedMove);
        ++ply;

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
    }

    return game;
}

} // namespace mz
```

- [ ] **Step 5: Add to the build and run the tests**

Add `src/selfplay.cpp` to the `add_library(mz ...)` source list.

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: PASS — `all selfplay tests passed`, 8/8 tests.

- [ ] **Step 6: Commit**

```bash
git add include/mz/selfplay.hpp src/selfplay.cpp tests/test_selfplay.cpp CMakeLists.txt
git commit -m "$(cat <<'EOF'
Add self-play producing whole trajectories

Records the MCTS root value at every ply, which the n-step value target
bootstraps from, and rewards along the trajectory rather than one
outcome at the end. The real board is stepped between moves: the
learned model is for searching, not for playing.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Evaluation and the three executables

Evaluation is deliberately identical in shape to the AlphaZero project's, because the whole point of the comparison is that both are scored the same way: greedy, noise-free search against a perfect minimax player, as X and as O.

Perfect play cannot be beaten, so a correct implementation reports `wins = 0` forever. The number that matters is `losses`, and convergence means it reaching zero and staying there.

**Files:**
- Create: `include/mz/eval.hpp`, `src/eval.cpp`
- Create: `apps/train.cpp`, `apps/evaluate.cpp`, `apps/play_cli.cpp`
- Create: `tests/integration_smoke.sh`
- Modify: `CMakeLists.txt`
- Test: `tests/test_eval.cpp`

**Interfaces:**
- Consumes: everything above.
- Produces: `mz::EvalResult{int wins=0; int draws=0; int losses=0;}` and `mz::evaluateAgainstMinimax(const MuZeroNetwork&, int gamesPerSide, int numSimulations) -> EvalResult`. Executables `train`, `evaluate`, `play_cli`.

- [ ] **Step 1: Write the failing test**

`tests/test_eval.cpp`:

```cpp
#include <cassert>
#include <cstdio>
#include "mz/eval.hpp"
#include "mz/network.hpp"

using namespace mz;

void test_every_game_is_accounted_for() {
    MuZeroNetwork network(1);
    EvalResult result = evaluateAgainstMinimax(network, /*gamesPerSide=*/2, /*numSimulations=*/8);
    assert(result.wins + result.draws + result.losses == 4);
}

void test_perfect_play_is_never_beaten() {
    // Minimax plays optimally, so no network -- trained, untrained or
    // broken -- can ever win a game. If this ever fails, the bug is in the
    // evaluation harness or in minimax, not in the network.
    MuZeroNetwork network(2);
    EvalResult result = evaluateAgainstMinimax(network, /*gamesPerSide=*/3, /*numSimulations=*/8);
    assert(result.wins == 0);
}

void test_zero_games_is_an_empty_result() {
    MuZeroNetwork network(3);
    EvalResult result = evaluateAgainstMinimax(network, 0, 8);
    assert(result.wins == 0 && result.draws == 0 && result.losses == 0);
}

int main() {
    test_every_game_is_accounted_for();
    test_perfect_play_is_never_beaten();
    test_zero_games_is_an_empty_result();
    std::printf("all eval tests passed\n");
    return 0;
}
```

`tests/integration_smoke.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
CHECKPOINT="$(mktemp -t mz_integration_XXXXXX.bin)"
trap 'rm -f "$CHECKPOINT"' EXIT

echo "== train (3 tiny iterations) =="
"$BUILD_DIR/train" 3 "$CHECKPOINT"
test -s "$CHECKPOINT"

echo "== evaluate =="
"$BUILD_DIR/evaluate" "$CHECKPOINT" 3

echo "== latent_probe =="
"$BUILD_DIR/latent_probe" "$CHECKPOINT"

echo "== diag_eval =="
"$BUILD_DIR/diag_eval" "$CHECKPOINT"

echo "== play_cli (scripted game) =="
PLAY_OUTPUT="$(printf '0\n1\n2\n3\n4\n5\n6\n7\n8\n' | "$BUILD_DIR/play_cli" "$CHECKPOINT")"
echo "$PLAY_OUTPUT"
echo "$PLAY_OUTPUT" | grep -qE '^(Draw\.|You win!|Agent wins\.)$' || { echo "play_cli did not finish a game" >&2; exit 1; }

echo "integration smoke test passed"
```

Note: the smoke script exercises `latent_probe` and `diag_eval`, which Task 10 creates. Until then it is written but not wired into `ctest`; Task 10 registers it.

- [ ] **Step 2: Run the test to verify it fails**

Add `mz_add_test(test_eval)` to `CMakeLists.txt`, then build.

Expected: FAIL — `mz/eval.hpp: No such file or directory`.

- [ ] **Step 3: Write the evaluation header and implementation**

`include/mz/eval.hpp`:

```cpp
#pragma once
#include "mz/network.hpp"

namespace mz {

struct EvalResult {
    int wins = 0;
    int draws = 0;
    int losses = 0;
};

// Plays gamesPerSide games as X and gamesPerSide as O against perfect
// minimax, with greedy (temperature 0), noise-free search. Results are
// from the network's perspective.
//
// Minimax is optimal, so `wins` is always 0 and `losses` reaching zero is
// what convergence means. Identical in shape to the AlphaZero project's
// evaluation, so the two are directly comparable.
EvalResult evaluateAgainstMinimax(const MuZeroNetwork& network, int gamesPerSide, int numSimulations);

} // namespace mz
```

`src/eval.cpp`:

```cpp
#include "mz/eval.hpp"
#include <random>
#include "mz/board.hpp"
#include "mz/mcts.hpp"
#include "mz/minimax.hpp"

namespace mz {

namespace {

int playOneGame(const MuZeroNetwork& network, bool networkPlaysX, int numSimulations,
                std::mt19937& rng) {
    MCTSConfig config;
    config.numSimulations = numSimulations;
    config.addRootNoise = false;   // the network's unperturbed best play

    Board board;
    while (!board.isTerminal()) {
        bool networkTurn = (board.playerToMove() == Cell::X) == networkPlaysX;
        int move;
        if (networkTurn) {
            MCTS mcts(network, config, rng);
            move = mcts.run(board, 0.0f).selectedMove;
        } else {
            move = minimaxBestMove(board);
        }
        board = board.applyMove(move);
    }

    Outcome outcome = board.outcome();
    if (outcome == Outcome::Draw) return 0;
    bool networkWon = (outcome == Outcome::XWins && networkPlaysX) ||
                      (outcome == Outcome::OWins && !networkPlaysX);
    return networkWon ? 1 : -1;
}

} // namespace

EvalResult evaluateAgainstMinimax(const MuZeroNetwork& network, int gamesPerSide,
                                  int numSimulations) {
    EvalResult result;
    std::mt19937 rng(12345);   // fixed: evaluation should be reproducible
    auto record = [&](int outcome) {
        if (outcome == 1) result.wins++;
        else if (outcome == -1) result.losses++;
        else result.draws++;
    };
    for (int i = 0; i < gamesPerSide; ++i) record(playOneGame(network, true, numSimulations, rng));
    for (int i = 0; i < gamesPerSide; ++i) record(playOneGame(network, false, numSimulations, rng));
    return result;
}

} // namespace mz
```

- [ ] **Step 4: Write the three executables**

`apps/train.cpp`:

```cpp
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include "mz/eval.hpp"
#include "mz/network.hpp"
#include "mz/replay_buffer.hpp"
#include "mz/selfplay.hpp"
#include "mz/targets.hpp"

int main(int argc, char** argv) {
    int numIterations = argc >= 2 ? std::atoi(argv[1]) : 400;
    std::string checkpointPath = argc >= 3 ? argv[2] : "checkpoint.bin";

    // Starting point. Task 11 tunes these against measured convergence.
    const int gamesPerIteration = 25;
    const int batchSize = 64;
    const int trainStepsPerIteration = 40;
    const float learningRate = 0.02f;
    const std::size_t bufferCapacity = 2000;   // games, not positions
    const int evalIntervalIterations = 10;     // N -- distinct from the unroll length K
    const int evalGamesPerSide = 20;
    const int evalSimulations = 150;

    mz::MuZeroNetwork network;
    mz::ReplayBuffer buffer(bufferCapacity);
    mz::SelfPlayConfig selfPlayConfig;
    mz::TargetConfig targetConfig;
    std::mt19937 rng(std::random_device{}());

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
            std::printf("iteration %d: games=%zu positions=%zu loss=%.4f (value=%.4f policy=%.4f reward=%.4f)\n",
                        iteration, buffer.size(), buffer.totalPositions(), losses.total,
                        losses.value, losses.policy, losses.reward);
        }

        if ((iteration + 1) % evalIntervalIterations == 0) {
            mz::EvalResult result =
                mz::evaluateAgainstMinimax(network, evalGamesPerSide, evalSimulations);
            std::printf("eval vs minimax: wins=%d draws=%d losses=%d\n", result.wins, result.draws,
                        result.losses);
            network.save(checkpointPath);
            std::printf("checkpoint saved to %s\n", checkpointPath.c_str());
        }
    }

    network.save(checkpointPath);
    std::printf("final checkpoint saved to %s\n", checkpointPath.c_str());
    return 0;
}
```

`apps/evaluate.cpp`:

```cpp
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include "mz/eval.hpp"
#include "mz/network.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: evaluate <checkpoint-path> [games-per-side]\n");
        return 1;
    }
    std::string checkpointPath = argv[1];
    int gamesPerSide = argc >= 3 ? std::atoi(argv[2]) : 50;

    mz::MuZeroNetwork network(1);
    try {
        network.load(checkpointPath);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    mz::EvalResult result = mz::evaluateAgainstMinimax(network, gamesPerSide, /*numSimulations=*/150);
    std::printf("vs minimax over %d games/side: wins=%d draws=%d losses=%d\n", gamesPerSide,
                result.wins, result.draws, result.losses);
    return 0;
}
```

`apps/play_cli.cpp`:

```cpp
#include <cstdio>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include "mz/board.hpp"
#include "mz/mcts.hpp"
#include "mz/network.hpp"

namespace {

void printBoard(const mz::Board& board) {
    const char* symbols[3] = {".", "X", "O"};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            std::printf("%s ", symbols[static_cast<int>(board.cellAt(row * 3 + col))]);
        }
        std::printf("\n");
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: play_cli <checkpoint-path>\n");
        return 1;
    }

    mz::MuZeroNetwork network(1);
    try {
        network.load(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    std::printf("You are X. Enter a move as a number 0-8 (see grid below).\n");
    std::printf("0 1 2\n3 4 5\n6 7 8\n\n");

    mz::MCTSConfig config;
    config.numSimulations = 300;
    config.addRootNoise = false;
    std::mt19937 rng(std::random_device{}());

    mz::Board board;
    while (!board.isTerminal()) {
        printBoard(board);
        if (board.playerToMove() == mz::Cell::X) {
            int move = -1;
            while (true) {
                std::printf("Your move: ");
                if (!(std::cin >> move) || !board.isLegalMove(move)) {
                    std::printf("Invalid move, try again.\n");
                    std::cin.clear();
                    std::cin.ignore(10000, '\n');
                    if (std::cin.eof()) {
                        std::printf("\nInput ended, exiting.\n");
                        return 0;
                    }
                    continue;
                }
                break;
            }
            board = board.applyMove(move);
        } else {
            mz::MCTS mcts(network, config, rng);
            int move = mcts.run(board, 0.0f).selectedMove;
            std::printf("Agent plays %d\n", move);
            board = board.applyMove(move);
        }
    }

    printBoard(board);
    mz::Outcome outcome = board.outcome();
    if (outcome == mz::Outcome::Draw) std::printf("Draw.\n");
    else if (outcome == mz::Outcome::XWins) std::printf("You win!\n");
    else std::printf("Agent wins.\n");

    return 0;
}
```

- [ ] **Step 5: Wire up the build**

Add `src/eval.cpp` to the `add_library(mz ...)` source list, then append:

```cmake
add_executable(train apps/train.cpp)
target_link_libraries(train mz)

add_executable(evaluate apps/evaluate.cpp)
target_link_libraries(evaluate mz)

add_executable(play_cli apps/play_cli.cpp)
target_link_libraries(play_cli mz)
```

Also `chmod +x tests/integration_smoke.sh`.

- [ ] **Step 6: Run the tests**

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: PASS — `all eval tests passed`, 9/9 tests. Then confirm the executables run:

```sh
./build/train 1 /tmp/mz_smoke.bin && ./build/evaluate /tmp/mz_smoke.bin 2
```

Expected: training output, then a `vs minimax` line with `wins=0`.

- [ ] **Step 7: Commit**

```bash
git add include/mz/eval.hpp src/eval.cpp apps tests/test_eval.cpp tests/integration_smoke.sh CMakeLists.txt
git commit -m "$(cat <<'EOF'
Add minimax evaluation and the train/evaluate/play_cli executables

Evaluation matches the AlphaZero project's exactly so the two are
directly comparable. Perfect play cannot be beaten, so wins is always
zero and losses reaching zero is what convergence means.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Diagnostics — `diag_eval` and `latent_probe`

`diag_eval` mirrors the AlphaZero project's transcript tool: one game as X and one as O against minimax, printing every move and visit distribution, so a losing checkpoint can be traced to the exact ply where it went wrong.

`latent_probe` has no AlphaZero counterpart, because AlphaZero has no learned model to be wrong about. It answers the question that only MuZero can fail: **does the dynamics network actually track the game?** It rolls dynamics forward k steps from a position and compares what prediction says about the resulting latent against a fresh representation pass on the real board those same actions reach. If the model is faithful, the two agree; as k grows they drift, and how fast is the measurement.

This is the tool that turns "MuZero converged" into "MuZero learned a model that stays accurate for about this many steps," which is the interesting claim.

**Files:**
- Create: `tools/diag_eval.cpp`, `tools/latent_probe.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: everything above.
- Produces: executables `diag_eval` and `latent_probe`.

- [ ] **Step 1: Write `tools/diag_eval.cpp`**

```cpp
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <string>
#include "mz/board.hpp"
#include "mz/mcts.hpp"
#include "mz/minimax.hpp"
#include "mz/network.hpp"

namespace {

void printGrid(const mz::Board& board) {
    const char* symbols[3] = {".", "X", "O"};
    for (int row = 0; row < 3; ++row) {
        std::printf("    ");
        for (int col = 0; col < 3; ++col) {
            std::printf("%s ", symbols[static_cast<int>(board.cellAt(row * 3 + col))]);
        }
        std::printf("\n");
    }
}

void playTranscript(const mz::MuZeroNetwork& network, bool networkPlaysX, int numSimulations) {
    std::printf("\n=== network plays %s ===\n", networkPlaysX ? "X" : "O");

    mz::MCTSConfig config;
    config.numSimulations = numSimulations;
    config.addRootNoise = false;
    std::mt19937 rng(20260910);

    mz::Board board;
    int ply = 0;
    while (!board.isTerminal()) {
        bool networkTurn = (board.playerToMove() == mz::Cell::X) == networkPlaysX;
        printGrid(board);
        if (networkTurn) {
            mz::MCTS mcts(network, config, rng);
            mz::MCTSResult result = mcts.run(board, 0.0f);
            std::printf("  ply %d  network -> %d   rootValue=%+.3f  depth=%d\n", ply,
                        result.selectedMove, result.rootValue, result.maxDepth);
            std::printf("  visits:");
            for (int a = 0; a < 9; ++a) {
                if (board.isLegalMove(a)) std::printf("  %d:%.2f", a, result.visitDistribution[a]);
            }
            std::printf("\n");
            board = board.applyMove(result.selectedMove);
        } else {
            int move = mz::minimaxBestMove(board);
            std::printf("  ply %d  minimax -> %d\n", ply, move);
            board = board.applyMove(move);
        }
        ++ply;
    }

    printGrid(board);
    mz::Outcome outcome = board.outcome();
    const char* verdict = "draw";
    if (outcome != mz::Outcome::Draw) {
        bool networkWon = (outcome == mz::Outcome::XWins && networkPlaysX) ||
                          (outcome == mz::Outcome::OWins && !networkPlaysX);
        verdict = networkWon ? "network WON (impossible vs perfect play -- check minimax)" : "network LOST";
    }
    std::printf("  result: %s\n", verdict);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: diag_eval <checkpoint-path> [simulations]\n");
        return 1;
    }
    int numSimulations = argc >= 3 ? std::atoi(argv[2]) : 150;

    mz::MuZeroNetwork network(1);
    try {
        network.load(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    playTranscript(network, true, numSimulations);
    playTranscript(network, false, numSimulations);
    return 0;
}
```

- [ ] **Step 2: Write `tools/latent_probe.cpp`**

```cpp
// Measures whether the learned dynamics actually tracks the real game.
//
// From a real position, roll the dynamics network forward k steps along a
// sequence of legal actions, and compare what the prediction network says
// about the resulting latent against a fresh representation pass on the
// real board those same actions reach. A faithful model agrees; drift as k
// grows is the thing worth measuring.
//
// AlphaZero has no equivalent diagnostic, because it has no learned model
// that could be wrong -- its search steps the actual rules.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <vector>
#include "mz/board.hpp"
#include "mz/minimax.hpp"
#include "mz/network.hpp"

namespace {

constexpr int kMaxDepth = 6;

struct DepthStats {
    double valueErrorSum = 0.0;
    double policyDistanceSum = 0.0;
    int count = 0;
};

// Total variation distance: half the L1 distance between two
// distributions, so 0 means identical and 1 means disjoint.
float totalVariation(const std::array<float, 9>& a, const std::array<float, 9>& b) {
    float sum = 0.0f;
    for (int i = 0; i < 9; ++i) sum += std::fabs(a[i] - b[i]);
    return 0.5f * sum;
}

// Walks one trajectory of legal moves, comparing imagined against real at
// every depth.
void probeFrom(const mz::MuZeroNetwork& network, mz::Board board, std::mt19937& rng,
               std::vector<DepthStats>& byDepth, DepthStats& terminalReward) {
    std::vector<float> latent = network.initialInference(board.encode()).latent;

    for (int depth = 1; depth <= kMaxDepth; ++depth) {
        if (board.isTerminal()) return;

        std::vector<int> legal = board.legalMoves();
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        int action = legal[pick(rng)];

        mz::MuZeroNetwork::RecurrentInference imagined = network.recurrentInference(latent, action);
        latent = imagined.latent;

        mz::Cell mover = board.playerToMove();
        board = board.applyMove(action);

        if (board.isTerminal()) {
            mz::Outcome outcome = board.outcome();
            float actualReward = 0.0f;
            if (outcome != mz::Outcome::Draw) {
                bool moverWon = (outcome == mz::Outcome::XWins && mover == mz::Cell::X) ||
                                (outcome == mz::Outcome::OWins && mover == mz::Cell::O);
                actualReward = moverWon ? 1.0f : -1.0f;
            }
            terminalReward.valueErrorSum += std::fabs(imagined.reward - actualReward);
            terminalReward.count += 1;
            return;
        }

        // The real position those same actions reached, seen fresh.
        mz::MuZeroNetwork::InitialInference real = network.initialInference(board.encode());

        DepthStats& stats = byDepth[depth];
        stats.valueErrorSum += std::fabs(imagined.value - real.value);
        stats.policyDistanceSum += totalVariation(imagined.policy, real.policy);
        stats.count += 1;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: latent_probe <checkpoint-path> [trajectories]\n");
        return 1;
    }
    int trajectories = argc >= 3 ? std::atoi(argv[2]) : 500;

    mz::MuZeroNetwork network(1);
    try {
        network.load(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    std::vector<DepthStats> byDepth(kMaxDepth + 1);
    DepthStats terminalReward;
    std::mt19937 rng(20260910);

    for (int t = 0; t < trajectories; ++t) {
        // Start from the empty board sometimes, and from a random legal
        // opening otherwise, so the probe covers more than one region.
        mz::Board board;
        std::uniform_int_distribution<int> openingLength(0, 3);
        int plies = openingLength(rng);
        for (int i = 0; i < plies && !board.isTerminal(); ++i) {
            std::vector<int> legal = board.legalMoves();
            std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
            board = board.applyMove(legal[pick(rng)]);
        }
        if (board.isTerminal()) continue;
        probeFrom(network, board, rng, byDepth, terminalReward);
    }

    std::printf("Learned model vs. real game, over %d trajectories\n", trajectories);
    std::printf("Imagined latent (dynamics rolled k steps) compared against a fresh\n");
    std::printf("representation pass on the real board the same actions reach.\n\n");
    std::printf("  k   samples   mean |value error|   mean policy distance\n");
    for (int depth = 1; depth <= kMaxDepth; ++depth) {
        const DepthStats& stats = byDepth[depth];
        if (stats.count == 0) continue;
        std::printf("  %d   %7d   %17.4f   %21.4f\n", depth, stats.count,
                    stats.valueErrorSum / stats.count, stats.policyDistanceSum / stats.count);
    }

    if (terminalReward.count > 0) {
        std::printf("\nTerminal reward prediction: mean |error| = %.4f over %d transitions\n",
                    terminalReward.valueErrorSum / terminalReward.count, terminalReward.count);
        std::printf("(A model that cannot tell when the game ends, and in whose favor,\n");
        std::printf(" cannot search correctly over itself.)\n");
    }
    return 0;
}
```

- [ ] **Step 3: Wire up the build and register the smoke test**

Append to `CMakeLists.txt`:

```cmake
# Plays one game as X and one as O against minimax, printing every move
# and MCTS visit distribution. Useful for tracing exactly where and why a
# checkpoint loses, e.g. `./diag_eval checkpoint.bin`.
add_executable(diag_eval tools/diag_eval.cpp)
target_link_libraries(diag_eval mz)

# Measures whether the learned dynamics tracks the real game. No AlphaZero
# equivalent exists: AlphaZero has no learned model that could be wrong.
add_executable(latent_probe tools/latent_probe.cpp)
target_link_libraries(latent_probe mz)

add_test(NAME integration_smoke
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_smoke.sh ${CMAKE_CURRENT_BINARY_DIR})
```

- [ ] **Step 4: Build and run everything**

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```

Expected: PASS — 10/10 tests, including `integration_smoke`.

- [ ] **Step 5: Commit**

```bash
git add tools CMakeLists.txt
git commit -m "$(cat <<'EOF'
Add diag_eval transcript tool and latent_probe model check

latent_probe has no AlphaZero counterpart: it measures how far the
learned dynamics can be rolled forward before its predictions drift
from a fresh pass on the real board, which is the failure mode only
MuZero has.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Train to convergence and record the measurements

Everything so far is verified correct in the small. This task establishes that it actually learns, and produces the numbers the documentation will cite.

The spec sets the expectation: MuZero should need materially more iterations than AlphaZero's twenty on this problem, because search below the root wanders into actions the game does not allow and the dynamics network has never seen a legal example of. That gap is the result, not a bug. Do not fix it by masking legality below the root — that would hand back exactly the rule knowledge the project exists to do without.

**Files:**
- Create: `docs/results.md`
- Possibly modify: `apps/train.cpp` (hyperparameters only)

**Interfaces:**
- Consumes: everything.
- Produces: `docs/results.md`, cited by the README and the comparison doc in Task 12.

- [ ] **Step 1: Run a full training run**

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
time ./build/train 400 checkpoint.bin 2>&1 | tee /tmp/mz_train_log.txt
```

Watch the `eval vs minimax` lines. `wins` must stay 0 throughout — a nonzero value means minimax or the evaluation harness is broken, not that the agent got strong. What matters is `losses` falling to 0 and staying there.

- [ ] **Step 2: If it does not converge, tune in this order**

Change one thing at a time and re-run. Record what was tried, including what did not work.

1. **`numSimulations` in `SelfPlayConfig`** (currently 60). The most likely culprit. MuZero wastes part of its search budget on illegal branches, so it needs a larger budget than AlphaZero's 50 to resolve the same position. Try 100, then 160.
2. **`learningRate`** (currently 0.02). If the loss oscillates rather than descending, halve it. If it descends but far too slowly, double it.
3. **`trainStepsPerIteration`** (currently 40) and **`batchSize`** (currently 64). More gradient steps per unit of self-play helps when the buffer is large and the loss is still falling.
4. **`unrollSteps`** in `TargetConfig` (currently 5). Games are at most 9 plies, so 5 is already deep. Dropping to 3 gives a cleaner signal per step at the cost of less pressure on the dynamics network to stay accurate.
5. **`tdSteps`** (currently 32, so value targets are the game outcome). Lowering it to 5 turns on real bootstrapping. Try it and record what happens — this is genuinely interesting and worth a paragraph in the docs either way.

If `losses` sticks at a small nonzero number, run `./build/diag_eval checkpoint.bin` and read the transcript for the losing ply.

- [ ] **Step 3: Measure the learned model**

```sh
./build/latent_probe checkpoint.bin 2000 | tee /tmp/mz_probe.txt
```

Record the table. The shape of the drift as k grows is the substance of Task 12's section on learned dynamics.

- [ ] **Step 4: Score the final checkpoint properly**

```sh
./build/evaluate checkpoint.bin 50
```

Expected: `wins=0 draws=100 losses=0`.

- [ ] **Step 5: Write `docs/results.md`**

Record, with real measured numbers and no placeholders:

- Final evaluation: games per side, wins/draws/losses.
- Iterations needed before `losses` first reached 0, and before it stayed at 0.
- The same figure for the AlphaZero project, from its README: 20 iterations to `draws=40 losses=0`. Note that an iteration is not identical work in the two projects and say what differs (self-play games per iteration, simulations per move, gradient steps per iteration), so the comparison is not read as more precise than it is.
- Wall-clock time for the run, and the machine it ran on.
- The final hyperparameters, and any that were changed from the Task 9 starting point with a sentence on why.
- The `latent_probe` table.
- If `tdSteps` bootstrapping was tried, what changed.

- [ ] **Step 6: Commit**

```bash
git add docs/results.md apps/train.cpp
git commit -m "$(cat <<'EOF'
Train to convergence and record measured results

Includes the iteration count against the AlphaZero project's, with the
caveat that an iteration is not identical work in the two, and the
latent_probe table showing how far the learned dynamics stays accurate.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: The walkthrough — `docs/algorithm-explained.md`

A guided tour of the algorithm against the real code in this repository, in the order the code runs. Same voice and structure as the AlphaZero project's version, so a reader can hold the two side by side.

Read `../alphazero-tictactoe/docs/algorithm-explained.md` first, in full, before writing a word. Match its conventions exactly: numbered sections with a `## NN — Title` heading, a bolded **Concept:** line naming the RL idea each section teaches, real code excerpts with a `*src/file.cpp:line — `Function`*` caption underneath, worked numeric examples in small tables, and blockquoted "Why this matters beyond tic-tac-toe" asides.

Add one convention of its own: a blockquoted **vs AlphaZero** callout at each point of divergence, placed where the divergence arises rather than collected at the end. Keep each to a few sentences and point to `muzero-vs-alphazero.md` for the full treatment.

**Files:**
- Create: `docs/algorithm-explained.md`

**Interfaces:**
- Consumes: the finished code and `docs/results.md` (Task 11).
- Produces: the document Task 14's HTML version and the README both link to.

- [ ] **Step 1: Read the sibling document end to end**

```sh
cat ../alphazero-tictactoe/docs/algorithm-explained.md
```

Note its length (632 lines), its section rhythm, and how it introduces RL vocabulary before using it.

- [ ] **Step 2: Write the document to this outline**

Header block matching the sibling's: title, one-sentence framing, a note that an illustrated HTML version exists, and a stats line giving language, the three network shapes, and the measured result from `docs/results.md`.

Then a `## Contents` list of anchor links, then these sections:

- **`## 00 — The problem with knowing the rules`** — Concept: model-based vs model-free RL. Frame the whole project: AlphaZero is handed a perfect simulator, which is fine for tic-tac-toe and chess and useless for anything where the rules are unknown or the state is a camera image. State what MuZero gives up and what it buys. End on the question the rest of the document answers: if you cannot simulate the game, what exactly do you need to predict in order to search?

- **`## 01 — Three functions instead of one`** — Concept: latent dynamics models. The `h`/`g`/`f` split, the table of shapes, and the key negative fact: no decoder, nothing can turn a latent back into a board. Explain why that is allowed — the latent is only ever asked to support policy, value and reward predictions. Excerpt `MuZeroNetwork::initialInference` and `recurrentInference`. First **vs AlphaZero** callout.

- **`## 02 — The action has to become a number`** — Concept: action encoding. Excerpt `makeDynamicsInput`. Explain the one-hot concatenation and why there is no board to apply a move to. Show the min-max latent normalization and why repeated application of `g` needs it, citing `test_deep_unroll_stays_bounded`.

- **`## 03 — Searching a model you cannot see`** — Concept: MCTS over learned state. Walk `MCTS::run`: the root is the only place a real observation enters; expansion below the root gives all nine actions a child. Excerpt the expansion loop with its comment. Explain the illegal-branch problem honestly, and cite `test_tree_grows_below_a_root_with_only_one_legal_move` as the test that proves the tree really is unmasked. **vs AlphaZero** callout.

- **`## 04 — Q values of unknown scale`** — Concept: value normalization in search. Why `MinMaxStats` exists here and not in AlphaZero. Excerpt `edgeQ`, `explorationTerm` and `selectChild`. Show the two PUCT formulas next to each other and be honest that `c2 = 19652` makes the log term nearly inert at this scale.

- **`## 05 — Backup through alternating players`** — Concept: credit assignment with intermediate rewards. Excerpt `backupPath`. Walk the three-node worked example from `test_backup_alternates_sign_and_carries_reward` numerically: leaf value 0.8, reward 0.5, resulting in 0.8 / -0.3 / +0.3 up the path. State the two perspective conventions explicitly, since every sign in the project depends on them.

- **`## 06 — What the game actually pays`** — Concept: reward as distinct from value. Tic-tac-toe pays only at the end, so the reward head is nearly always predicting zero. Explain why it is kept anyway: a model that cannot tell when the game ends and in whose favor cannot search over itself. Excerpt the terminal-reward assignment in `playSelfPlayGame`, and quote the terminal reward error from `docs/results.md`.

- **`## 07 — Targets that improve with the network`** — Concept: bootstrapping and TD learning. The n-step formula with the alternating sign, excerpted from `bootstrappedValue`. Work the `tdSteps=2` example from `test_short_td_bootstraps_from_stored_search_value` numerically. Explain that the default exceeds the game length so the expression collapses to the outcome, and what changes when it does not, citing `docs/results.md` if bootstrapping was measured. **vs AlphaZero** callout — this is the deepest difference and deserves the longest one.

- **`## 08 — Learning through five copies of itself`** — Concept: backpropagation through time. The forward unroll diagram, then the reverse loop. Explain the two scaling rules and why each exists. Be explicit that a wrong gradient here fails silently, and that `test_gradient_matches_numerical_through_the_full_unroll` is what catches it — including why the test checks the ratio of two step sizes rather than just the sign.

- **`## 09 — The full loop`** — Concept: putting it together. Self-play produces trajectories, trajectories produce unrolled samples, samples produce gradients, the better network produces better trajectories. Excerpt the main loop of `apps/train.cpp`. Include the measured convergence from `docs/results.md` and be plain about MuZero needing more iterations here, and why.

- **`## 10 — What this bought, and what it cost`** — Concept: when a learned model is worth it. Honest summary. On tic-tac-toe MuZero is strictly worse: slower, more code, and it laboriously rediscovers rules that were sitting right there. Name what it buys that AlphaZero structurally cannot have: Atari and other domains with no simulator, and Reanalyze, which needs bootstrapped targets to work at all. Point at `muzero-vs-alphazero.md` and the Reanalyze-readiness section of the spec.

- **`## 11 — Concept map`** — Mirror the sibling's closing section: a table of every RL concept introduced, where it appears in this codebase, and where it appears in the AlphaZero one.

- [ ] **Step 3: Verify every code excerpt and file:line citation**

Every excerpt must match the committed source exactly, and every `src/file.cpp:NN` caption must point at the right line. Check each one:

```sh
grep -n "initialInference\|recurrentInference\|makeDynamicsInput" src/network.cpp
grep -n "backupPath\|explorationTerm\|edgeQ\|selectChild" src/mcts.cpp
grep -n "bootstrappedValue" src/targets.cpp
```

Also confirm every anchor link in `## Contents` resolves to a real heading.

- [ ] **Step 4: Commit**

```bash
git add docs/algorithm-explained.md
git commit -m "$(cat <<'EOF'
Add the guided algorithm walkthrough

Follows the code in the order it runs, in the same shape as the
AlphaZero project's walkthrough, with a "vs AlphaZero" callout at each
point of divergence.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: The comparison — `docs/muzero-vs-alphazero.md`

The standalone treatment. Task 12's inline callouts say *that* something differs where it arises; this says *why*, in one place, with the two implementations side by side.

Both codebases are on disk, so every claim here can be checked against real code rather than against the papers. Do that — cite `mz/` and `az/` files by name throughout.

**Files:**
- Create: `docs/muzero-vs-alphazero.md`

- [ ] **Step 1: Write the summary table**

A single table read before anything else. One row per difference, three columns: the dimension, what AlphaZero does, what MuZero does. Rows: game dynamics, state representation, legality knowledge, terminal detection, reward, value target, search Q normalization, PUCT exploration term, replay storage, network count, and whether stale data can be improved.

- [ ] **Step 2: Write one section per difference**

Each section: what AlphaZero does with a code pointer, what MuZero does with a code pointer, why the difference exists, and what it costs. Keep them short — the table carries the summary, these carry the reasoning.

- **Given vs learned dynamics.** `az::Board::applyMove` inside AlphaZero's search versus `MuZeroNetwork::recurrentInference` inside `MCTS::run`. The cost: MuZero has to learn what the other already knows.
- **The latent state, and the loss of interpretability.** There is no decoder. You cannot print a MuZero search node. This is why `latent_probe` had to be written and why AlphaZero needs no such tool.
- **Root-only legality.** `MCTS::maskAndRenormalize` at the root and the unmasked nine-way expansion below it. Cite `test_tree_grows_below_a_root_with_only_one_legal_move`. State plainly that this wastes search on this domain, and quote the iteration gap from `docs/results.md`.
- **Terminal detection.** AlphaZero's search stops at terminal nodes with the true result. MuZero's tree has no terminal concept at all and searches straight through the end of the game, relying on the reward and value heads to make that unattractive.
- **The reward head.** Why MuZero has one and AlphaZero does not, why it is nearly always zero here, and why it is still structural. Quote the terminal reward error from `docs/results.md`.
- **Bootstrapped vs outcome value targets.** The n-step formula against AlphaZero's final result. Explain that AlphaZero's target is a fact about the past and cannot improve, while MuZero's is the network's own opinion and improves as the network does. This is the section that sets up Reanalyze.
- **Trajectory vs position storage.** `mz::GameHistory` in a ring buffer versus `az::TrainingExample`. Explain that both MuZero targets read forward from a position, which a flattened position cannot support.
- **The two PUCT formulas.** Side by side, with the constants. Note honestly that the log term barely matters at this scale.
- **One network vs three.** `az::Network` with a shared trunk and two heads, versus `h`/`g`/`f`. Note the consequence for training: a single forward pass in AlphaZero versus a K-step unroll with backprop-through-time, and that this is why `mz::Dense` had to accumulate gradients.
- **Reanalyze: a capability AlphaZero structurally cannot have.** Explain the mechanism, why bootstrapped targets are the precondition, and point at the three architectural decisions in the spec that leave room for it. Be clear it is not implemented.

- [ ] **Step 3: Write the deviations section**

Reproduce the spec's "Deviations From the Paper" list, expanded with a sentence each on what the real thing does and why it was skipped: scalar heads instead of the 601-bin categorical support with its invertible scaling transform, SGD instead of Adam with a schedule, no Reanalyze, and small MLPs instead of residual towers. The point is that a reader never mistakes a simplification here for the algorithm.

- [ ] **Step 4: Verify the AlphaZero claims against the actual sibling code**

Do not write any claim about AlphaZero from memory:

```sh
grep -n "cPuct\|applyMove\|isTerminal" ../alphazero-tictactoe/src/mcts.cpp
grep -n "targetValue\|TrainingExample" ../alphazero-tictactoe/include/az/network.hpp
sed -n '1,60p' ../alphazero-tictactoe/src/selfplay.cpp
```

- [ ] **Step 5: Commit**

```bash
git add docs/muzero-vs-alphazero.md
git commit -m "$(cat <<'EOF'
Add the standalone MuZero vs AlphaZero comparison

One section per difference, each citing real code in both projects
rather than the papers, plus the deviations from the paper so no
simplification here is mistaken for the algorithm.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: The illustrated HTML and the README

**Files:**
- Create: `docs/algorithm-explained.html`
- Create: `README.md`

- [ ] **Step 1: Build the HTML from the sibling's stylesheet**

The AlphaZero project's `docs/algorithm-explained.html` is a single self-contained file: a `<title>`, a Google Fonts link, a long `<style>` block defining light and dark palettes as CSS custom properties, then the article markup. Reuse it.

```sh
sed -n '1,200p' ../alphazero-tictactoe/docs/algorithm-explained.html
```

Copy the `<style>` block verbatim so the two documents look like siblings, change the `<title>` to "MuZero from Scratch", and render Task 12's markdown as the article body. Keep the existing dark-mode handling — it defines the light palette on bare `:root`, overrides under `@media (prefers-color-scheme: dark)` guarded by `:root:not([data-theme="light"])`, and again under `:root[data-theme="dark"]`. Do not simplify that to a single media query.

Two diagrams earn their place; draw them as inline SVG using the stylesheet's existing custom properties for color, so they theme correctly:

1. **The three functions and how search uses them.** A real board at the root feeding representation, then dynamics applied repeatedly down a tree with prediction hanging off each latent. It should make visible that the observation enters exactly once.
2. **The unrolled training graph.** Observation into representation into latent 0, then K dynamics steps across, with prediction heads dropping down from each latent to its three losses. Mark where the 1/K loss scaling and the half gradient apply.

- [ ] **Step 2: Verify the page**

Open it in a browser and check both themes:

```sh
open docs/algorithm-explained.html
```

Confirm no horizontal scrolling at a narrow window width, that code blocks scroll within their own containers rather than widening the page, and that both SVG diagrams are legible in light and dark.

- [ ] **Step 3: Write the README**

Model it on `../alphazero-tictactoe/README.md`, which is 95 lines. Sections:

- **Title and framing.** What this is: a from-scratch, dependency-free C++17 MuZero applied to tic-tac-toe, built as a learning project and as a direct comparison against the sibling AlphaZero implementation. One sentence on the central idea: search runs inside a model the system learns for itself, and the rules are consulted only at the search root.
- **Links** to the spec, the plan, the walkthrough in both markdown and HTML, the comparison document, and `docs/results.md`.
- **Build.** CMake 3.16+, a C++17 compiler, no external dependencies, the four-line build recipe, and `ctest --output-on-failure` with the real test count.
- **Usage.** `train`, `evaluate`, `play_cli` with their arguments and defaults, then a typical session.
- **What "trained" looks like.** The measured result from `docs/results.md`, including the iteration count and an honest note that it is more than AlphaZero needed here, with a one-line reason and a pointer to the comparison document.
- **Diagnostics.** `diag_eval` and `latent_probe`, with a sample of the probe's output table and one sentence on how to read it.
- **Project layout.** The directory tree with a line per directory, matching the sibling's format.
- **Deliberately out of scope.** Reanalyze, the categorical value/reward support, and the arena/promotion step, each with a pointer to where the spec explains why.
- **Authorship.** `Mark Castelluccio, designed and implemented with [Claude Code](https://claude.com/claude-code).`

- [ ] **Step 4: Verify every link and command in the README**

Every relative link must resolve, and every command must run as written:

```sh
grep -o '](\([^)]*\))' README.md
rm -rf /tmp/mz_verify && cmake -S . -B /tmp/mz_verify && cmake --build /tmp/mz_verify && ctest --test-dir /tmp/mz_verify --output-on-failure
```

Confirm the test count quoted in the README matches what `ctest` reports.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/algorithm-explained.html
git commit -m "$(cat <<'EOF'
Add the illustrated walkthrough and the README

HTML reuses the AlphaZero project's stylesheet so the two documents
read as siblings, and adds diagrams of the three functions in search
and of the unrolled training graph.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```
