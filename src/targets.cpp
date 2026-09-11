#include "mz/targets.hpp"
#include <cassert>
#include <cmath>

namespace mz {

namespace {

// The n-step bootstrapped value at absolute ply `t`, in the perspective of
// the player to move at `t`.
//
//   z_t = sum_{i in [t, t+n)} (-1)^(i-t) * discount^(i-t) * reward_i
//         + (-1)^n * discount^n * searchValue_{t+n}
//
// The alternating sign is the two-player part: rewards and search values
// are each stored in the perspective of whoever was to move at their own
// ply, so every ply of separation flips the sign relative to t.
float bootstrappedValue(const GameHistory& game, int t, const TargetConfig& config) {
    const int length = static_cast<int>(game.length());
    if (t >= length) return 0.0f;   // absorbing state past the end of the game

    float value = 0.0f;

    const int bootstrapIndex = t + config.tdSteps;
    if (bootstrapIndex < length) {
        float sign = (config.tdSteps % 2 == 0) ? 1.0f : -1.0f;
        value += sign * std::pow(config.discount, static_cast<float>(config.tdSteps)) *
                 game.searchValues[bootstrapIndex];
    }

    const int last = bootstrapIndex < length ? bootstrapIndex : length;
    for (int i = t; i < last; ++i) {
        float sign = ((i - t) % 2 == 0) ? 1.0f : -1.0f;
        value += sign * std::pow(config.discount, static_cast<float>(i - t)) * game.rewards[i];
    }

    return value;
}

std::array<float, 9> uniformPolicy() {
    std::array<float, 9> policy{};
    policy.fill(1.0f / 9.0f);
    return policy;
}

} // namespace

UnrolledSample makeUnrolledSample(const GameHistory& game, int position, const TargetConfig& config) {
    assert(position >= 0);
    assert(position < static_cast<int>(game.length()));
    assert(config.unrollSteps >= 0);

    const int length = static_cast<int>(game.length());
    const int K = config.unrollSteps;

    UnrolledSample sample;
    sample.observation = game.observations[position];
    sample.actions.resize(K);
    sample.targetValues.resize(K + 1);
    sample.targetRewards.assign(K + 1, 0.0f);
    sample.targetPolicies.resize(K + 1);

    for (int k = 0; k <= K; ++k) {
        const int t = position + k;

        sample.targetValues[k] = bootstrappedValue(game, t, config);

        // The reward for unroll step k is the one paid on the transition
        // INTO step k, i.e. out of ply t-1. Its perspective is already
        // that of the player who acted, which is what the reward head
        // predicts, so no sign adjustment is needed.
        if (k > 0) {
            const int from = t - 1;
            sample.targetRewards[k] = (from < length) ? game.rewards[from] : 0.0f;
        }

        sample.targetPolicies[k] = (t < length) ? game.searchPolicies[t] : uniformPolicy();

        if (k < K) {
            if (t < length) {
                sample.actions[k] = game.actions[t];
            } else {
                // Absorbing state. The paper samples a uniformly random
                // action here; a fixed cycle is used instead so the
                // function stays pure and the targets stay reproducible.
                // With the default unroll length K, k ranges over
                // [0, K), so this only ever emits actions 0..K-1 --
                // dynamics never sees actions K..8 in absorbing states.
                sample.actions[k] = k % 9;
            }
        }
    }

    return sample;
}

} // namespace mz
