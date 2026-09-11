#include "mz/board.hpp"
#include <cassert>

namespace mz {

namespace {
constexpr int kLines[8][3] = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
    {0, 4, 8}, {2, 4, 6},
};

// The lines through each square, terminated by -1. A move can only
// complete a line it is part of, so placing a stone needs to check at most
// four lines instead of all eight.
constexpr int kLinesThrough[9][5] = {
    { 0,  3,  6, -1, -1},   // square 0
    { 0,  4, -1, -1, -1},   // square 1
    { 0,  5,  7, -1, -1},   // square 2
    { 1,  3, -1, -1, -1},   // square 3
    { 1,  4,  6,  7, -1},   // square 4
    { 1,  5, -1, -1, -1},   // square 5
    { 2,  3,  7, -1, -1},   // square 6
    { 2,  4, -1, -1, -1},   // square 7
    { 2,  5,  6, -1, -1},   // square 8
};
}

Board::Board() : toMove_(Cell::X), outcome_(Outcome::Ongoing), filled_(0) {
    cells_.fill(Cell::Empty);
}

Cell Board::cellAt(int index) const {
    return cells_[index];
}

Cell Board::playerToMove() const {
    return toMove_;
}

bool Board::isLegalMove(int index) const {
    if (index < 0 || index >= 9) return false;
    if (isTerminal()) return false;
    return cells_[index] == Cell::Empty;
}

std::vector<int> Board::legalMoves() const {
    std::vector<int> moves;
    if (isTerminal()) return moves;
    for (int i = 0; i < 9; ++i) {
        if (cells_[i] == Cell::Empty) moves.push_back(i);
    }
    return moves;
}

Board Board::applyMove(int index) const {
    assert(isLegalMove(index));
    Board next = *this;
    next.cells_[index] = toMove_;
    next.toMove_ = (toMove_ == Cell::X) ? Cell::O : Cell::X;
    next.filled_ = static_cast<int8_t>(filled_ + 1);
    next.updateOutcomeAfter(index, toMove_);
    return next;
}

void Board::updateOutcomeAfter(int index, Cell mover) {
    for (const int* id = kLinesThrough[index]; *id != -1; ++id) {
        const int* line = kLines[*id];
        if (cells_[line[0]] == mover && cells_[line[1]] == mover && cells_[line[2]] == mover) {
            outcome_ = (mover == Cell::X) ? Outcome::XWins : Outcome::OWins;
            return;
        }
    }
    outcome_ = (filled_ == 9) ? Outcome::Draw : Outcome::Ongoing;
}

std::array<float, 18> Board::encode() const {
    std::array<float, 18> out{};
    Cell mine = toMove_;
    Cell theirs = (toMove_ == Cell::X) ? Cell::O : Cell::X;
    for (int i = 0; i < 9; ++i) {
        out[i] = (cells_[i] == mine) ? 1.0f : 0.0f;
        out[9 + i] = (cells_[i] == theirs) ? 1.0f : 0.0f;
    }
    return out;
}

bool Board::operator==(const Board& other) const {
    return cells_ == other.cells_ && toMove_ == other.toMove_;
}

} // namespace mz
