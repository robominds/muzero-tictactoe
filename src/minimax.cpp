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
    int best = std::numeric_limits<int>::min();
    for (int m : board.legalMoves()) {
        int childScore = -scoreOf(board.applyMove(m));
        if (childScore > best) best = childScore;
    }
    return best;
}

} // namespace

int minimaxBestMove(const Board& board) {
    assert(!board.isTerminal());
    int bestMove = board.legalMoves().front();
    int bestScore = std::numeric_limits<int>::min();
    for (int m : board.legalMoves()) {
        int score = -scoreOf(board.applyMove(m));
        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
        }
    }
    return bestMove;
}

} // namespace mz
