#include <cassert>
#include <cstdio>
#include "mz/board.hpp"
#include "mz/minimax.hpp"

using namespace mz;

void test_minimax_takes_immediate_win() {
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(3); // O
    b = b.applyMove(1); // X: X at 0,1, threat at 2
    b = b.applyMove(4); // O
    assert(minimaxBestMove(b) == 2);
}

void test_minimax_blocks_immediate_loss() {
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(5); // O
    b = b.applyMove(1); // X at 0,1 threatens 2 -- O to move must block
    assert(minimaxBestMove(b) == 2);
}

void test_minimax_never_loses_against_itself() {
    Board b;
    while (!b.isTerminal()) {
        b = b.applyMove(minimaxBestMove(b));
    }
    assert(b.outcome() == Outcome::Draw);
}

int main() {
    test_minimax_takes_immediate_win();
    test_minimax_blocks_immediate_loss();
    test_minimax_never_loses_against_itself();
    std::printf("all minimax tests passed\n");
    return 0;
}
