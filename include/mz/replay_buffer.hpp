#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include "mz/game_history.hpp"

namespace mz {

// Fixed-capacity ring buffer of whole trajectories -- not of positions,
// which is what AlphaZero's equivalent stores.
class ReplayBuffer {
public:
    struct Sample {
        std::size_t gameIndex;
        int position;
    };

    // seed drives only the internal sampling RNG (samplePositions), for
    // reproducible training runs. Defaults to a random seed, as before,
    // when not given.
    explicit ReplayBuffer(std::size_t capacity, std::uint32_t seed = std::random_device{}());

    void add(GameHistory game);

    std::size_t size() const { return games_.size(); }
    std::size_t totalPositions() const;

    // Precondition: index < size() (asserted).
    const GameHistory& game(std::size_t index) const;

    // Uniform over stored positions, with replacement: a game contributes
    // in proportion to its length, so long games are not under-sampled.
    // Precondition: totalPositions() > 0.
    std::vector<Sample> samplePositions(std::size_t count) const;

    // Reanalyze hook. Replaces a stored trajectory's search targets in
    // place, leaving observations, actions and rewards alone -- those are
    // facts about what happened and never go stale. Search policies and
    // values are the network's opinions at the time, and a stronger
    // network can improve them. Precondition: sizes match the trajectory
    // length (asserted).
    void replaceSearchTargets(std::size_t gameIndex,
                              std::vector<std::array<float, 9>> policies,
                              std::vector<float> values);

private:
    std::size_t capacity_;
    std::size_t nextIndex_ = 0;
    std::vector<GameHistory> games_;
    mutable std::mt19937 rng_;
};

} // namespace mz
