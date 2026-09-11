#include "mz/minimax.hpp"
#include <cassert>
#include <limits>

namespace mz {

namespace {

// Value of `board` from the perspective of board.playerToMove(), assuming
// optimal play by both sides: +1 win, -1 loss, 0 draw.
int scoreOf(const Board& board) {
    if (board.isTerminal()) {
        return board.outcome() == Outcome::Draw ? 0 : -1;
    }
    // Iterating squares directly rather than calling legalMoves(): this
    // recursion visits the whole game tree, and a fresh std::vector per
    // node was a measurable share of evaluation time. The ascending order
    // is the same one legalMoves() produces, so the result is unchanged.
    int best = std::numeric_limits<int>::min();
    for (int m = 0; m < 9; ++m) {
        if (!board.isLegalMove(m)) continue;
        int childScore = -scoreOf(board.applyMove(m));
        if (childScore > best) best = childScore;
    }
    return best;
}

} // namespace

int minimaxBestMove(const Board& board) {
    assert(!board.isTerminal());
    int bestMove = -1;
    int bestScore = std::numeric_limits<int>::min();
    for (int m = 0; m < 9; ++m) {
        if (!board.isLegalMove(m)) continue;
        if (bestMove < 0) bestMove = m;   // lowest legal index, as before
        int score = -scoreOf(board.applyMove(m));
        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
        }
    }
    assert(bestMove >= 0);
    return bestMove;
}

} // namespace mz
