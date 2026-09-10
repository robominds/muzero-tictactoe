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
