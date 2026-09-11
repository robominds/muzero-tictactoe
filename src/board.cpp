#include "mz/board.hpp"
#include <cassert>

namespace mz {

namespace {
constexpr int kLines[8][3] = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
    {0, 4, 8}, {2, 4, 6},
};
}

Board::Board() : toMove_(Cell::X), outcome_(Outcome::Ongoing) {
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
    next.refreshOutcome();
    return next;
}

void Board::refreshOutcome() {
    for (const auto& line : kLines) {
        Cell a = cells_[line[0]], b = cells_[line[1]], c = cells_[line[2]];
        if (a != Cell::Empty && a == b && b == c) {
            outcome_ = (a == Cell::X) ? Outcome::XWins : Outcome::OWins;
            return;
        }
    }
    for (int i = 0; i < 9; ++i) {
        if (cells_[i] == Cell::Empty) {
            outcome_ = Outcome::Ongoing;
            return;
        }
    }
    outcome_ = Outcome::Draw;
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
