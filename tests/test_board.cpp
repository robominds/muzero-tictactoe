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

int main() {
    test_new_board_has_nine_legal_moves();
    test_apply_move_alternates_player();
    test_row_win_detected();
    test_diagonal_win_detected();
    test_draw_detected();
    test_illegal_move_rejected_by_isLegalMove();
    test_encode_is_from_perspective_of_player_to_move();
    std::printf("all board tests passed\n");
    return 0;
}
