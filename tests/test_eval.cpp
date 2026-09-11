#include <cassert>
#include <cstdio>
#include "mz/eval.hpp"
#include "mz/network.hpp"

using namespace mz;

void test_every_game_is_accounted_for() {
    MuZeroNetwork network(1);
    EvalResult result = evaluateAgainstMinimax(network, /*gamesPerSide=*/2, /*numSimulations=*/8);
    assert(result.wins + result.draws + result.losses == 4);
}

void test_perfect_play_is_never_beaten() {
    // Minimax plays optimally, so no network -- trained, untrained or
    // broken -- can ever win a game. If this ever fails, the bug is in the
    // evaluation harness or in minimax, not in the network.
    MuZeroNetwork network(2);
    EvalResult result = evaluateAgainstMinimax(network, /*gamesPerSide=*/3, /*numSimulations=*/8);
    assert(result.wins == 0);
}

void test_zero_games_is_an_empty_result() {
    MuZeroNetwork network(3);
    EvalResult result = evaluateAgainstMinimax(network, 0, 8);
    assert(result.wins == 0 && result.draws == 0 && result.losses == 0);
}

int main() {
    test_every_game_is_accounted_for();
    test_perfect_play_is_never_beaten();
    test_zero_games_is_an_empty_result();
    std::printf("all eval tests passed\n");
    return 0;
}
