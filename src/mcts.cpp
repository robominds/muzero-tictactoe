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
    // One root plus nine children per expanded node, and one node is
    // expanded per simulation. Reserving up front means a search in steady
    // state never touches the allocator.
    nodes_.reserve(static_cast<std::size_t>(config_.numSimulations) * 9 + 16);
}

int MCTS::acquireNode() {
    if (nodeCount_ == nodes_.size()) nodes_.emplace_back();
    Node& node = nodes_[nodeCount_];
    node.stats = BackupNode{};
    node.prior = 0.0f;
    node.expanded = false;
    node.children.fill(-1);
    // node.latent deliberately keeps its capacity; it is overwritten
    // before it is read.
    return static_cast<int>(nodeCount_++);
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

int MCTS::selectChild(int parentIndex, const MinMaxStats& stats) const {
    const Node& parent = nodes_[parentIndex];
    float bestScore = -std::numeric_limits<float>::infinity();
    int bestAction = -1;
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
    assert(bestAction >= 0);
    return bestAction;
}

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

    MinMaxStats stats;
    int nodesExpanded = 0;
    int maxDepth = 0;

    for (int sim = 0; sim < config_.numSimulations; ++sim) {
        path_.clear();
        path_.push_back(rootIndex);
        int nodeIndex = rootIndex;
        int lastAction = -1;

        while (nodes_[nodeIndex].expanded) {
            lastAction = selectChild(nodeIndex, stats);
            nodeIndex = nodes_[nodeIndex].children[lastAction];
            path_.push_back(nodeIndex);
        }

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

        // Taken only now that the arena has finished growing for this
        // simulation, so these pointers stay valid through backup.
        statsPath_.clear();
        statsPath_.reserve(path_.size());
        for (int index : path_) statsPath_.push_back(&nodes_[index].stats);
        backupPath(statsPath_, step.value, kDiscount, stats);
    }

    MCTSResult result{};
    result.visitDistribution.fill(0.0f);
    result.nodesExpanded = nodesExpanded;
    result.maxDepth = maxDepth;
    result.rootValue = nodeValue(nodes_[rootIndex].stats);

    std::array<float, 9> counts{};
    counts.fill(0.0f);
    float totalVisits = 0.0f;
    for (int a = 0; a < 9; ++a) {
        if (nodes_[rootIndex].children[a] < 0) continue;
        counts[a] = static_cast<float>(nodes_[nodes_[rootIndex].children[a]].stats.visitCount);
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
            if (draw < running) {
                chosen = a;
                break;
            }
        }
        result.selectedMove = chosen;
    }

    return result;
}

} // namespace mz
