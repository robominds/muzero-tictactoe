#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include "mz/board.hpp"
#include "mz/network.hpp"
#include "mz/selfplay.hpp"

using namespace mz;

namespace {
bool near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) < tol; }

GameHistory playOne(unsigned seed) {
    MuZeroNetwork network(seed);
    SelfPlayConfig config;
    config.numSimulations = 12;   // small: these tests check structure, not strength
    std::mt19937 rng(seed);
    return playSelfPlayGame(network, config, rng);
}
} // namespace

void test_game_length_is_plausible() {
    GameHistory game = playOne(1);
    assert(game.length() >= 5 && game.length() <= 9);
}

void test_all_trajectory_arrays_have_the_same_length() {
    GameHistory game = playOne(2);
    assert(game.observations.size() == game.length());
    assert(game.searchPolicies.size() == game.length());
    assert(game.searchValues.size() == game.length());
    assert(game.rewards.size() == game.length());
}

void test_replaying_the_actions_reproduces_the_trajectory() {
    // The strongest structural check: every recorded action was legal in
    // the position whose observation was recorded alongside it, and the
    // sequence really does end the game.
    GameHistory game = playOne(3);
    Board board;
    for (std::size_t t = 0; t < game.length(); ++t) {
        assert(!board.isTerminal());
        auto encoded = board.encode();
        for (int i = 0; i < 18; ++i) assert(near(encoded[i], game.observations[t][i]));
        assert(board.isLegalMove(game.actions[t]));
        board = board.applyMove(game.actions[t]);
    }
    assert(board.isTerminal());
}

void test_rewards_are_zero_until_the_final_move() {
    GameHistory game = playOne(4);
    for (std::size_t t = 0; t + 1 < game.length(); ++t) assert(near(game.rewards[t], 0.0f));
}

void test_final_reward_matches_the_outcome() {
    // Whoever plays the last move either wins or draws -- in tic-tac-toe
    // you cannot lose by moving. So the final reward is +1 or 0.
    GameHistory game = playOne(5);
    Board board;
    for (int action : game.actions) board = board.applyMove(action);
    float finalReward = game.rewards.back();
    if (board.outcome() == Outcome::Draw) assert(near(finalReward, 0.0f));
    else assert(near(finalReward, 1.0f));
}

void test_search_policies_never_recommend_an_occupied_square() {
    GameHistory game = playOne(6);
    Board board;
    for (std::size_t t = 0; t < game.length(); ++t) {
        float sum = 0.0f;
        for (int a = 0; a < 9; ++a) {
            if (!board.isLegalMove(a)) assert(game.searchPolicies[t][a] == 0.0f);
            sum += game.searchPolicies[t][a];
        }
        assert(near(sum, 1.0f, 1e-4f));
        board = board.applyMove(game.actions[t]);
    }
}

void test_search_values_are_recorded_and_finite() {
    // AlphaZero stores nothing like this. Here every ply's root value is
    // kept, because it is what the n-step value target bootstraps from.
    GameHistory game = playOne(7);
    for (float v : game.searchValues) {
        assert(std::isfinite(v));
        assert(v >= -2.0f && v <= 2.0f);
    }
}

void test_two_seeds_produce_different_games() {
    GameHistory a = playOne(8);
    GameHistory b = playOne(9);
    assert(a.actions != b.actions || a.length() != b.length());
}

int main() {
    test_game_length_is_plausible();
    test_all_trajectory_arrays_have_the_same_length();
    test_replaying_the_actions_reproduces_the_trajectory();
    test_rewards_are_zero_until_the_final_move();
    test_final_reward_matches_the_outcome();
    test_search_policies_never_recommend_an_occupied_square();
    test_search_values_are_recorded_and_finite();
    test_two_seeds_produce_different_games();
    std::printf("all selfplay tests passed\n");
    return 0;
}
