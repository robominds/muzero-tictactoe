#pragma once
#include <random>
#include "mz/game_history.hpp"
#include "mz/network.hpp"

namespace mz {

struct SelfPlayConfig {
    // Task 11 tuning (see docs/results.md): raised from 60 to 100. Tried
    // 160 and 300 too; both did as well as or worse than 100 across
    // matched seeds -- more search budget is not simply better once most
    // of it is already being spent below the point where any legal game
    // could still be running (see the search-depth finding in
    // docs/results.md). 100 is an empirical middle ground, not a value
    // with a principled derivation.
    int numSimulations = 100;
    // Plies from the start that use temperature=1.0 sampling, for training
    // data diversity. Greedy afterward.
    //
    // Task 11 tuning (see docs/results.md): raised from 2. AlphaZero's
    // search steps the real rules below the root, so its greedy tail
    // (plies >= this) reliably rediscovers a corrective line if the
    // network is ever wrong about it. MuZero's greedy tail searches over
    // its own learned dynamics, which drifts from the real game the
    // deeper it is rolled (see the latent_probe table) -- so once a
    // MuZero self-play game goes fully greedy, a wrong-but-confident
    // belief a few plies deep can persist indefinitely because self-play
    // never plays the position that would correct it. Sampling for more
    // plies buys more chances for that position to actually occur in
    // training data.
    int temperatureMoves = 6;
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
