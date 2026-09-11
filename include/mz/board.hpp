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

    // Cached, not recomputed. outcome() is called constantly -- isTerminal()
    // calls it, and legalMoves()/isLegalMove() call isTerminal() -- so
    // rescanning all eight winning lines each time showed up as a
    // meaningful share of training time. It is computed once per move, in
    // applyMove(), and read back here.
    Outcome outcome() const { return outcome_; }
    bool isTerminal() const { return outcome_ != Outcome::Ongoing; }

    // 18 floats: [my stones (9), opponent stones (9)], from the
    // perspective of playerToMove(). MuZero sees this only at the search
    // root -- everywhere else it works with latents produced from it.
    std::array<float, 18> encode() const;

    bool operator==(const Board& other) const;

private:
    // Updates outcome_ after `index` was just played by `mover`. Only the
    // lines through that square can have newly completed, so this checks
    // four of them rather than rescanning all eight.
    void updateOutcomeAfter(int index, Cell mover);

    std::array<Cell, 9> cells_;
    Cell toMove_;
    // Both derived from cells_, so neither takes part in operator==.
    Outcome outcome_;
    int8_t filled_;
};

} // namespace mz
