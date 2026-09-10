#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <string>
#include "mz/board.hpp"
#include "mz/network.hpp"
#include "mz/targets.hpp"

using namespace mz;

namespace {

std::string tempPath() {
    return std::string("mz_test_checkpoint_") + std::to_string(std::rand()) + ".bin";
}

bool near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) < tol; }

} // namespace

void test_initial_inference_shapes_and_ranges() {
    MuZeroNetwork network(1234);
    Board board;
    auto out = network.initialInference(board.encode());

    assert(out.latent.size() == MuZeroNetwork::kLatentSize);
    // Latents are min-max normalized into [0, 1] so repeated dynamics
    // applications cannot let magnitudes drift.
    for (float v : out.latent) assert(v >= -1e-6f && v <= 1.0f + 1e-6f);

    float sum = 0.0f;
    for (float p : out.policy) { assert(p > 0.0f); sum += p; }
    assert(near(sum, 1.0f));
    assert(out.value >= -1.0f && out.value <= 1.0f);
}

void test_policy_head_is_not_masked_to_legal_moves() {
    // The network knows nothing about legality. Masking happens in MCTS,
    // and only at the root.
    MuZeroNetwork network(99);
    Board board;
    board = board.applyMove(4);
    auto out = network.initialInference(board.encode());
    assert(out.policy[4] > 0.0f);   // an illegal move, still given a prior
}

void test_recurrent_inference_shapes_and_ranges() {
    MuZeroNetwork network(555);
    Board board;
    auto initial = network.initialInference(board.encode());
    auto next = network.recurrentInference(initial.latent, 4);

    assert(next.latent.size() == MuZeroNetwork::kLatentSize);
    for (float v : next.latent) assert(v >= -1e-6f && v <= 1.0f + 1e-6f);
    assert(next.reward >= -1.0f && next.reward <= 1.0f);
    assert(next.value >= -1.0f && next.value <= 1.0f);
    float sum = 0.0f;
    for (float p : next.policy) sum += p;
    assert(near(sum, 1.0f));
}

void test_recurrent_inference_depends_on_the_action() {
    // The action must actually reach the dynamics network. If the one-hot
    // were dropped or misaligned, every action would produce the same
    // successor and search would be meaningless.
    MuZeroNetwork network(7);
    Board board;
    auto initial = network.initialInference(board.encode());
    auto a = network.recurrentInference(initial.latent, 0);
    auto b = network.recurrentInference(initial.latent, 8);

    bool differs = false;
    for (int i = 0; i < MuZeroNetwork::kLatentSize; ++i) {
        if (std::fabs(a.latent[i] - b.latent[i]) > 1e-5f) differs = true;
    }
    assert(differs);
}

void test_recurrent_inference_is_deterministic() {
    MuZeroNetwork network(31);
    Board board;
    auto initial = network.initialInference(board.encode());
    auto a = network.recurrentInference(initial.latent, 3);
    auto b = network.recurrentInference(initial.latent, 3);
    for (int i = 0; i < MuZeroNetwork::kLatentSize; ++i) assert(near(a.latent[i], b.latent[i]));
    assert(near(a.reward, b.reward));
}

void test_deep_unroll_stays_bounded() {
    // Apply dynamics far deeper than search ever will. Min-max
    // normalization must keep latents in range no matter how many times
    // the recurrence is applied.
    MuZeroNetwork network(17);
    Board board;
    auto state = network.initialInference(board.encode());
    std::vector<float> latent = state.latent;
    for (int step = 0; step < 50; ++step) {
        auto next = network.recurrentInference(latent, step % 9);
        latent = next.latent;
        for (float v : latent) {
            assert(std::isfinite(v));
            assert(v >= -1e-6f && v <= 1.0f + 1e-6f);
        }
        assert(std::isfinite(next.value));
        assert(std::isfinite(next.reward));
    }
}

void test_different_seeds_give_different_networks() {
    // Not the empty board: Dense initializes biases to zero and only seeds
    // its weights, so an all-zero observation forces a bias-only forward
    // pass through every layer -- value 0 and uniform policy for every
    // seed, by construction, regardless of the weights. A board with a
    // stone on it is the one that actually exercises the seeded weights.
    MuZeroNetwork a(1);
    MuZeroNetwork b(2);
    Board board;
    board = board.applyMove(4);
    assert(!near(a.initialInference(board.encode()).value,
                 b.initialInference(board.encode()).value));
}

void test_save_load_round_trips() {
    MuZeroNetwork original(4242);
    std::string path = tempPath();
    original.save(path);

    MuZeroNetwork restored(1);
    restored.load(path);
    std::remove(path.c_str());

    Board board;
    board = board.applyMove(0);
    board = board.applyMove(4);
    auto a = original.initialInference(board.encode());
    auto b = restored.initialInference(board.encode());

    assert(near(a.value, b.value));
    for (int i = 0; i < 9; ++i) assert(near(a.policy[i], b.policy[i]));
    for (int i = 0; i < MuZeroNetwork::kLatentSize; ++i) assert(near(a.latent[i], b.latent[i]));

    auto ra = original.recurrentInference(a.latent, 2);
    auto rb = restored.recurrentInference(b.latent, 2);
    assert(near(ra.reward, rb.reward));
    assert(near(ra.value, rb.value));
}

void test_load_rejects_a_non_checkpoint_file() {
    std::string path = tempPath();
    FILE* f = std::fopen(path.c_str(), "wb");
    std::fputs("this is not a checkpoint", f);
    std::fclose(f);

    MuZeroNetwork network(1);
    bool threw = false;
    try {
        network.load(path);
    } catch (const std::exception&) {
        threw = true;
    }
    std::remove(path.c_str());
    assert(threw);
}

void test_load_reports_a_missing_file() {
    MuZeroNetwork network(1);
    bool threw = false;
    try {
        network.load("definitely_not_a_real_path_12345.bin");
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw);
}

int main() {
    test_initial_inference_shapes_and_ranges();
    test_policy_head_is_not_masked_to_legal_moves();
    test_recurrent_inference_shapes_and_ranges();
    test_recurrent_inference_depends_on_the_action();
    test_recurrent_inference_is_deterministic();
    test_deep_unroll_stays_bounded();
    test_different_seeds_give_different_networks();
    test_save_load_round_trips();
    test_load_rejects_a_non_checkpoint_file();
    test_load_reports_a_missing_file();
    std::printf("all network tests passed\n");
    return 0;
}
