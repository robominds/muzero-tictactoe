#include <cassert>
#include <cstdio>
#include <set>
#include "mz/replay_buffer.hpp"

using namespace mz;

namespace {

// A trajectory of `length` moves, tagged so tests can tell games apart:
// every searchValue is `tag`.
GameHistory makeGame(int length, float tag) {
    GameHistory game;
    for (int t = 0; t < length; ++t) {
        std::array<float, 18> obs{};
        obs[0] = static_cast<float>(t);
        game.observations.push_back(obs);
        game.actions.push_back(t % 9);
        std::array<float, 9> policy{};
        policy.fill(1.0f / 9.0f);
        game.searchPolicies.push_back(policy);
        game.searchValues.push_back(tag);
        game.rewards.push_back(0.0f);
    }
    return game;
}

} // namespace

void test_length_is_number_of_moves() {
    GameHistory game = makeGame(5, 0.0f);
    assert(game.length() == 5);
    assert(game.observations.size() == 5);
    assert(game.searchValues.size() == 5);
    assert(game.rewards.size() == 5);
}

void test_add_and_size() {
    ReplayBuffer buffer(10);
    assert(buffer.size() == 0);
    buffer.add(makeGame(5, 1.0f));
    buffer.add(makeGame(7, 2.0f));
    assert(buffer.size() == 2);
    assert(buffer.totalPositions() == 12);
}

void test_ring_evicts_oldest_at_capacity() {
    ReplayBuffer buffer(3);
    for (int i = 0; i < 5; ++i) buffer.add(makeGame(2, static_cast<float>(i)));
    assert(buffer.size() == 3);
    // Games 0 and 1 were evicted; 2, 3, 4 remain in some slot.
    std::set<float> tags;
    for (size_t i = 0; i < buffer.size(); ++i) tags.insert(buffer.game(i).searchValues[0]);
    assert(tags == (std::set<float>{2.0f, 3.0f, 4.0f}));
    assert(buffer.totalPositions() == 6);
}

void test_sampled_positions_are_in_range() {
    ReplayBuffer buffer(10);
    buffer.add(makeGame(4, 1.0f));
    buffer.add(makeGame(9, 2.0f));
    auto samples = buffer.samplePositions(200);
    assert(samples.size() == 200);
    for (const auto& s : samples) {
        assert(s.gameIndex < buffer.size());
        assert(s.position >= 0);
        assert(s.position < static_cast<int>(buffer.game(s.gameIndex).length()));
    }
}

void test_sampling_reaches_both_games() {
    ReplayBuffer buffer(10);
    buffer.add(makeGame(5, 1.0f));
    buffer.add(makeGame(5, 2.0f));
    auto samples = buffer.samplePositions(500);
    std::set<size_t> seen;
    for (const auto& s : samples) seen.insert(s.gameIndex);
    assert(seen.size() == 2);
}

void test_replace_search_targets_overwrites_in_place() {
    // The hook Reanalyze needs: refresh a stored trajectory's search
    // targets with the results of a newer network, leaving observations,
    // actions and rewards untouched.
    ReplayBuffer buffer(10);
    buffer.add(makeGame(3, 1.0f));

    std::vector<std::array<float, 9>> policies(3);
    for (auto& p : policies) { p.fill(0.0f); p[4] = 1.0f; }
    std::vector<float> values = {0.5f, -0.5f, 0.25f};
    buffer.replaceSearchTargets(0, policies, values);

    const GameHistory& game = buffer.game(0);
    assert(game.length() == 3);
    assert(game.searchValues[1] == -0.5f);
    assert(game.searchPolicies[2][4] == 1.0f);
    assert(game.observations[2][0] == 2.0f);   // untouched
    assert(game.actions[1] == 1);              // untouched
}

int main() {
    test_length_is_number_of_moves();
    test_add_and_size();
    test_ring_evicts_oldest_at_capacity();
    test_sampled_positions_are_in_range();
    test_sampling_reaches_both_games();
    test_replace_search_targets_overwrites_in_place();
    std::printf("all replay buffer tests passed\n");
    return 0;
}
