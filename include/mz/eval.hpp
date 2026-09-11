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
