#include <cassert>
#include <cmath>
#include <cstdio>
#include "mz/targets.hpp"

using namespace mz;

namespace {

// X wins on move 4 (the fifth ply). Search values are distinctive so the
// bootstrap term is identifiable; rewards are zero until the final move.
//   ply:            0     1     2     3     4
//   player:         X     O     X     O     X
//   searchValue:  0.10 -0.20  0.30 -0.40  0.50
//   reward:       0     0     0     0     +1  (to X, who just moved)
GameHistory makeWinForFirstPlayer() {
    GameHistory game;
    float values[5] = {0.10f, -0.20f, 0.30f, -0.40f, 0.50f};
    for (int t = 0; t < 5; ++t) {
        std::array<float, 18> obs{};
        obs[0] = static_cast<float>(t);
        game.observations.push_back(obs);
        game.actions.push_back(t);
        std::array<float, 9> policy{};
        policy.fill(0.0f);
        policy[t] = 1.0f;                    // one-hot, easy to identify
        game.searchPolicies.push_back(policy);
        game.searchValues.push_back(values[t]);
        game.rewards.push_back(t == 4 ? 1.0f : 0.0f);
    }
    return game;
}

bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }

} // namespace

void test_array_sizes_follow_unroll_length() {
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 3;
    UnrolledSample sample = makeUnrolledSample(game, 0, config);
    assert(sample.actions.size() == 3);
    assert(sample.targetValues.size() == 4);
    assert(sample.targetRewards.size() == 4);
    assert(sample.targetPolicies.size() == 4);
    assert(sample.targetRewards[0] == 0.0f);   // step 0 has no transition in
}

void test_observation_and_actions_come_from_position() {
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 2;
    UnrolledSample sample = makeUnrolledSample(game, 1, config);
    assert(sample.observation[0] == 1.0f);
    assert(sample.actions[0] == 1);
    assert(sample.actions[1] == 2);
}

void test_long_td_collapses_to_game_outcome() {
    // tdSteps beyond the game length: the bootstrap falls off the end, so
    // the value target is just the discounted, sign-flipped reward sum --
    // the plain game outcome, which is exactly AlphaZero's target.
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 4;
    config.tdSteps = 32;
    UnrolledSample sample = makeUnrolledSample(game, 0, config);

    // X wins. From ply 0 (X to move) that is +1; from ply 1 (O) -1; etc.
    assert(near(sample.targetValues[0], 1.0f));
    assert(near(sample.targetValues[1], -1.0f));
    assert(near(sample.targetValues[2], 1.0f));
    assert(near(sample.targetValues[3], -1.0f));
}

void test_short_td_bootstraps_from_stored_search_value() {
    // tdSteps = 2 from ply 0: no reward in [0, 2), so the target is the
    // search value at ply 2, sign-flipped by (2 - 0) plies = even = +1.
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 1;
    config.tdSteps = 2;
    UnrolledSample sample = makeUnrolledSample(game, 0, config);
    assert(near(sample.targetValues[0], 0.30f));

    // From ply 1 the bootstrap lands on ply 3, again an even gap: -0.40.
    UnrolledSample fromOne = makeUnrolledSample(game, 1, config);
    assert(near(fromOne.targetValues[0], -0.40f));
}

void test_odd_td_gap_flips_the_sign() {
    // tdSteps = 1 from ply 0 bootstraps on ply 1, an odd gap, so the stored
    // value (-0.20, from O's perspective) is negated into X's.
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 1;
    config.tdSteps = 1;
    UnrolledSample sample = makeUnrolledSample(game, 0, config);
    assert(near(sample.targetValues[0], 0.20f));
}

void test_reward_targets_align_with_incoming_transition() {
    // Unrolling 4 steps from ply 1 reaches plies 1..5. The only nonzero
    // reward is on the transition out of ply 4, which arrives at step 4.
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 4;
    UnrolledSample sample = makeUnrolledSample(game, 1, config);
    assert(sample.targetRewards[0] == 0.0f);
    assert(sample.targetRewards[1] == 0.0f);   // out of ply 1
    assert(sample.targetRewards[2] == 0.0f);   // out of ply 2
    assert(sample.targetRewards[3] == 0.0f);   // out of ply 3
    assert(near(sample.targetRewards[4], 1.0f));   // out of ply 4: X wins
}

void test_policy_targets_come_from_stored_search() {
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 2;
    UnrolledSample sample = makeUnrolledSample(game, 2, config);
    assert(near(sample.targetPolicies[0][2], 1.0f));
    assert(near(sample.targetPolicies[1][3], 1.0f));
    assert(near(sample.targetPolicies[2][4], 1.0f));
}

void test_absorbing_padding_past_end_of_game() {
    // Unrolling 3 steps from ply 3 runs off the end of a 5-ply game at
    // step 2. Past terminal: zero value, zero reward, uniform policy.
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 3;
    UnrolledSample sample = makeUnrolledSample(game, 3, config);

    assert(near(sample.targetValues[2], 0.0f));
    assert(near(sample.targetValues[3], 0.0f));
    assert(near(sample.targetRewards[3], 0.0f));
    for (int a = 0; a < 9; ++a) assert(near(sample.targetPolicies[2][a], 1.0f / 9.0f));
    for (int a = 0; a < 9; ++a) assert(near(sample.targetPolicies[3][a], 1.0f / 9.0f));
}

void test_padding_actions_cycle_deterministically() {
    // The paper samples random actions past terminal. This uses a fixed
    // cycle instead: deterministic and reproducible. With unrollSteps=4
    // here, k ranges over [0, 4), so this only ever emits actions 0..3 --
    // it does not spread across all nine actions.
    GameHistory game = makeWinForFirstPlayer();
    TargetConfig config;
    config.unrollSteps = 4;
    UnrolledSample a = makeUnrolledSample(game, 3, config);
    UnrolledSample b = makeUnrolledSample(game, 3, config);
    assert(a.actions == b.actions);
    for (int action : a.actions) assert(action >= 0 && action < 9);
}

void test_draw_gives_zero_value_targets_everywhere() {
    GameHistory game;
    for (int t = 0; t < 9; ++t) {
        game.observations.push_back(std::array<float, 18>{});
        game.actions.push_back(t);
        std::array<float, 9> policy{};
        policy.fill(1.0f / 9.0f);
        game.searchPolicies.push_back(policy);
        game.searchValues.push_back(0.0f);
        game.rewards.push_back(0.0f);
    }
    TargetConfig config;
    config.unrollSteps = 5;
    UnrolledSample sample = makeUnrolledSample(game, 0, config);
    for (float v : sample.targetValues) assert(near(v, 0.0f));
    for (float r : sample.targetRewards) assert(near(r, 0.0f));
}

int main() {
    test_array_sizes_follow_unroll_length();
    test_observation_and_actions_come_from_position();
    test_long_td_collapses_to_game_outcome();
    test_short_td_bootstraps_from_stored_search_value();
    test_odd_td_gap_flips_the_sign();
    test_reward_targets_align_with_incoming_transition();
    test_policy_targets_come_from_stored_search();
    test_absorbing_padding_past_end_of_game();
    test_padding_actions_cycle_deterministically();
    test_draw_gives_zero_value_targets_everywhere();
    std::printf("all target tests passed\n");
    return 0;
}
