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
