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
