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
