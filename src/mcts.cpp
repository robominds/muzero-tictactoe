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
    MuZeroNetwork::InitialInference initial = network_.initialInference(board.encode(), workspace_);
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
            network_.recurrentInference(parent->latent, lastAction, workspace_);

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
