#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace mz {

enum class Cell : int8_t { Empty = 0, X = 1, O = 2 };

enum class Outcome { Ongoing, XWins, OWins, Draw };

class Board {
public:
    Board();

    Cell cellAt(int index) const;
    Cell playerToMove() const;

    bool isLegalMove(int index) const;
    std::vector<int> legalMoves() const;

    // Precondition: isLegalMove(index) == true (asserted).
    Board applyMove(int index) const;

    Outcome outcome() const;
    bool isTerminal() const { return outcome() != Outcome::Ongoing; }

    // 18 floats: [my stones (9), opponent stones (9)], from the
    // perspective of playerToMove(). MuZero sees this only at the search
    // root -- everywhere else it works with latents produced from it.
    std::array<float, 18> encode() const;

    bool operator==(const Board& other) const;

private:
    std::array<Cell, 9> cells_;
    Cell toMove_;
};

} // namespace mz
