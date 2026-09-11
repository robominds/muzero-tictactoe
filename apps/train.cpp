#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include "mz/eval.hpp"
#include "mz/network.hpp"
#include "mz/replay_buffer.hpp"
#include "mz/selfplay.hpp"
#include "mz/targets.hpp"

int main(int argc, char** argv) {
    int numIterations = argc >= 2 ? std::atoi(argv[1]) : 400;
    std::string checkpointPath = argc >= 3 ? argv[2] : "checkpoint.bin";

    // Starting point. Task 11 tunes these against measured convergence.
    const int gamesPerIteration = 25;
    const int batchSize = 64;
    const int trainStepsPerIteration = 40;
    const float learningRate = 0.02f;
    const std::size_t bufferCapacity = 2000;   // games, not positions
    const int evalIntervalIterations = 10;     // N -- distinct from the unroll length K
    const int evalGamesPerSide = 20;
    const int evalSimulations = 150;

    mz::MuZeroNetwork network;
    mz::ReplayBuffer buffer(bufferCapacity);
    mz::SelfPlayConfig selfPlayConfig;
    mz::TargetConfig targetConfig;
    std::mt19937 rng(std::random_device{}());

    for (int iteration = 0; iteration < numIterations; ++iteration) {
        for (int g = 0; g < gamesPerIteration; ++g) {
            buffer.add(mz::playSelfPlayGame(network, selfPlayConfig, rng));
        }

        if (buffer.totalPositions() >= static_cast<std::size_t>(batchSize)) {
            mz::MuZeroNetwork::Losses losses;
            for (int step = 0; step < trainStepsPerIteration; ++step) {
                std::vector<mz::UnrolledSample> batch;
                batch.reserve(batchSize);
                for (const auto& sample : buffer.samplePositions(batchSize)) {
                    batch.push_back(mz::makeUnrolledSample(buffer.game(sample.gameIndex),
                                                           sample.position, targetConfig));
                }
                losses = network.trainStep(batch, learningRate);
            }
            std::printf("iteration %d: games=%zu positions=%zu loss=%.4f (value=%.4f policy=%.4f reward=%.4f)\n",
                        iteration, buffer.size(), buffer.totalPositions(), losses.total,
                        losses.value, losses.policy, losses.reward);
        }

        if ((iteration + 1) % evalIntervalIterations == 0) {
            mz::EvalResult result =
                mz::evaluateAgainstMinimax(network, evalGamesPerSide, evalSimulations);
            std::printf("eval vs minimax: wins=%d draws=%d losses=%d\n", result.wins, result.draws,
                        result.losses);
            network.save(checkpointPath);
            std::printf("checkpoint saved to %s\n", checkpointPath.c_str());
        }
    }

    network.save(checkpointPath);
    std::printf("final checkpoint saved to %s\n", checkpointPath.c_str());
    return 0;
}
