#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include "mz/eval.hpp"
#include "mz/network.hpp"
#include "mz/replay_buffer.hpp"
#include "mz/selfplay.hpp"
#include "mz/targets.hpp"

namespace {

void printUsage() {
    std::fprintf(stderr,
                 "usage: train [iterations] [checkpoint-path] [seed] [resume-from]\n"
                 "\n"
                 "  iterations       training iterations to run (default 400)\n"
                 "  checkpoint-path  where to write checkpoints (default checkpoint.bin)\n"
                 "  seed             pin for a reproducible run; random if omitted\n"
                 "  resume-from      an existing checkpoint to start from, instead of\n"
                 "                   fresh random weights\n");
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0)) {
        printUsage();
        return 0;
    }

    int numIterations = argc >= 2 ? std::atoi(argv[1]) : 400;
    std::string checkpointPath = argc >= 3 ? argv[2] : "checkpoint.bin";
    // Optional: pin the seed for reproducible runs (e.g. a seed sweep).
    // Defaults to a random seed, as before, when not given.
    unsigned int seed = argc >= 4 ? static_cast<unsigned int>(std::strtoul(argv[3], nullptr, 10))
                                  : std::random_device{}();
    // Optional: continue from an existing checkpoint's weights.
    std::string resumeFrom = argc >= 5 ? argv[4] : "";

    // Starting point, tuned against measured convergence (see docs/results.md).
    const int gamesPerIteration = 25;
    const int batchSize = 64;
    const int trainStepsPerIteration = 40;
    const float learningRate = 0.02f;
    const std::size_t bufferCapacity = 2000;   // games, not positions
    const int evalIntervalIterations = 10;     // N -- distinct from the unroll length K
    const int evalGamesPerSide = 20;
    const int evalSimulations = 150;

    // Same seed drives both the network's initial weights and self-play/
    // training randomness, so a run is fully reproducible from one number.
    mz::MuZeroNetwork network(seed);
    if (!resumeFrom.empty()) {
        try {
            network.load(resumeFrom);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
    }
    mz::ReplayBuffer buffer(bufferCapacity, seed);
    mz::SelfPlayConfig selfPlayConfig;
    mz::TargetConfig targetConfig;
    std::mt19937 rng(seed);
    std::printf("seed=%u\n", seed);
    if (!resumeFrom.empty()) {
        // Weights only. Trajectories are not checkpointed, so a resumed run
        // begins with an empty replay buffer and spends its first
        // iterations refilling it -- expect the loss to jump before it
        // settles. The seed still fully determines the run, so resuming
        // twice from the same checkpoint with the same seed is
        // reproducible, but a resumed run is NOT identical to an
        // uninterrupted one of the same total length.
        std::printf("resumed from %s (weights only; replay buffer starts empty)\n",
                    resumeFrom.c_str());
    }

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
