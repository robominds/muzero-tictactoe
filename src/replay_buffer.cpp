#include "mz/replay_buffer.hpp"
#include <cassert>

namespace mz {

ReplayBuffer::ReplayBuffer(std::size_t capacity) : capacity_(capacity) {
    assert(capacity > 0);
    games_.reserve(capacity);
}

void ReplayBuffer::add(GameHistory game) {
    if (games_.size() < capacity_) {
        games_.push_back(std::move(game));
    } else {
        games_[nextIndex_] = std::move(game);
        nextIndex_ = (nextIndex_ + 1) % capacity_;
    }
}

std::size_t ReplayBuffer::totalPositions() const {
    std::size_t total = 0;
    for (const auto& game : games_) total += game.length();
    return total;
}

const GameHistory& ReplayBuffer::game(std::size_t index) const {
    assert(index < games_.size());
    return games_[index];
}

std::vector<ReplayBuffer::Sample> ReplayBuffer::samplePositions(std::size_t count) const {
    assert(totalPositions() > 0);
    // Draw a game with probability proportional to its length, then a
    // uniform position within it: equivalent to drawing uniformly over all
    // stored positions.
    std::vector<std::size_t> cumulative;
    cumulative.reserve(games_.size());
    std::size_t running = 0;
    for (const auto& game : games_) {
        running += game.length();
        cumulative.push_back(running);
    }

    std::uniform_int_distribution<std::size_t> dist(0, running - 1);
    std::vector<Sample> samples;
    samples.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t draw = dist(rng_);
        std::size_t gameIndex = 0;
        while (cumulative[gameIndex] <= draw) ++gameIndex;
        std::size_t before = gameIndex == 0 ? 0 : cumulative[gameIndex - 1];
        samples.push_back(Sample{gameIndex, static_cast<int>(draw - before)});
    }
    return samples;
}

void ReplayBuffer::replaceSearchTargets(std::size_t gameIndex,
                                        std::vector<std::array<float, 9>> policies,
                                        std::vector<float> values) {
    assert(gameIndex < games_.size());
    GameHistory& game = games_[gameIndex];
    assert(policies.size() == game.length());
    assert(values.size() == game.length());
    game.searchPolicies = std::move(policies);
    game.searchValues = std::move(values);
}

} // namespace mz
