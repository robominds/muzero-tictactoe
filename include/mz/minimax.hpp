#pragma once
#include "mz/board.hpp"

namespace mz {

// Returns the game-theoretically optimal move for board.playerToMove(),
// via exhaustive minimax search. Precondition: !board.isTerminal().
// Used only to measure convergence -- never for training, and never
// inside MCTS.
int minimaxBestMove(const Board& board);

} // namespace mz
