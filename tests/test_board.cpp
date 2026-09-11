#include <cassert>
#include <cstdio>
#include "mz/board.hpp"

using namespace mz;

void test_new_board_has_nine_legal_moves() {
    Board b;
    assert(b.legalMoves().size() == 9);
    assert(b.playerToMove() == Cell::X);
    assert(!b.isTerminal());
}

void test_apply_move_alternates_player() {
    Board b;
    Board b2 = b.applyMove(0);
    assert(b2.cellAt(0) == Cell::X);
    assert(b2.playerToMove() == Cell::O);
    assert(b2.legalMoves().size() == 8);
}

void test_row_win_detected() {
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(3); // O
    b = b.applyMove(1); // X
    b = b.applyMove(4); // O
    b = b.applyMove(2); // X completes top row
    assert(b.outcome() == Outcome::XWins);
    assert(b.isTerminal());
    assert(b.legalMoves().empty());
}

void test_diagonal_win_detected() {
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(1); // O
    b = b.applyMove(4); // X
    b = b.applyMove(2); // O
    b = b.applyMove(8); // X completes diagonal
    assert(b.outcome() == Outcome::XWins);
}

void test_draw_detected() {
    Board b;
    int moves[] = {0, 1, 2, 4, 3, 5, 7, 6, 8};
    for (int m : moves) b = b.applyMove(m);
    assert(b.outcome() == Outcome::Draw);
}

void test_illegal_move_rejected_by_isLegalMove() {
    Board b;
    b = b.applyMove(0);
    assert(!b.isLegalMove(0));
    assert(b.isLegalMove(1));
}

void test_encode_is_from_perspective_of_player_to_move() {
    Board b;
    b = b.applyMove(0); // X at 0, O to move
    auto enc = b.encode();
    for (int i = 0; i < 9; ++i) assert(enc[i] == 0.0f);
    assert(enc[9 + 0] == 1.0f);
}

// A full rescan of all eight lines, independent of Board's incremental
// bookkeeping. The reference the exhaustive test below compares against.
Outcome rescanOutcome(const Board& b) {
    static const int lines[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
        {0, 4, 8}, {2, 4, 6},
    };
    for (const auto& line : lines) {
        Cell a = b.cellAt(line[0]);
        if (a != Cell::Empty && a == b.cellAt(line[1]) && a == b.cellAt(line[2])) {
            return a == Cell::X ? Outcome::XWins : Outcome::OWins;
        }
    }
    for (int i = 0; i < 9; ++i) {
        if (b.cellAt(i) == Cell::Empty) return Outcome::Ongoing;
    }
    return Outcome::Draw;
}

int positionsChecked = 0;

// Walks every reachable position and checks the cached, incrementally
// maintained outcome against a from-scratch rescan.
//
// Board updates outcome_ in applyMove by checking only the lines through
// the square just played, using a hand-written table of which lines pass
// through which square. A single wrong entry in that table is invisible
// until some specific position is misjudged -- which is exactly what
// happened while this optimization was being written, and what this test
// now catches.
void walk(const Board& b) {
    ++positionsChecked;
    assert(b.outcome() == rescanOutcome(b));
    if (b.isTerminal()) return;
    for (int m : b.legalMoves()) walk(b.applyMove(m));
}

void test_cached_outcome_matches_a_full_rescan_everywhere() {
    Board b;
    walk(b);
    // 549946 nodes in the full tic-tac-toe game tree, counting repeats via
    // distinct move orders. Pinned so a search that silently stops early
    // cannot make this test vacuous.
    assert(positionsChecked == 549946);
}

int main() {
    test_new_board_has_nine_legal_moves();
    test_apply_move_alternates_player();
    test_row_win_detected();
    test_diagonal_win_detected();
    test_draw_detected();
    test_illegal_move_rejected_by_isLegalMove();
    test_encode_is_from_perspective_of_player_to_move();
    test_cached_outcome_matches_a_full_rescan_everywhere();
    std::printf("all board tests passed\n");
    return 0;
}
