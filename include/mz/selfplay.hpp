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
