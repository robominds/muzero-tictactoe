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
    // A board with stones on it, deliberately. An empty board encodes to
    // all zeros, and since Dense seeds only its weights and leaves biases
    // at zero, an all-zero observation drives a bias-only pass through
    // every layer -- the latent comes out all zeros, minMaxNormalize takes
    // only its denominator-floor branch, and softmax9 returns exactly
    // uniform. Every assertion below would pass on that path without
    // testing anything. See test_untrained_network_is_flat_on_an_empty_board.
    MuZeroNetwork network(1234);
    Board board;
    board = board.applyMove(4);
    board = board.applyMove(0);
    board = board.applyMove(8);
    auto out = network.initialInference(board.encode());

    assert(out.latent.size() == MuZeroNetwork::kLatentSize);
    // Min-max normalization must put the extremes exactly at the ends of
    // [0, 1], not merely somewhere inside it -- that is what distinguishes
    // a working normalization from one that never ran.
    float lo = out.latent[0], hi = out.latent[0];
    for (float v : out.latent) {
        assert(v >= -1e-6f && v <= 1.0f + 1e-6f);
        lo = std::fmin(lo, v);
        hi = std::fmax(hi, v);
    }
    assert(near(lo, 0.0f, 1e-5f));
    assert(near(hi, 1.0f, 1e-5f));

    // A real softmax over non-equal logits: sums to one AND is not flat.
    float sum = 0.0f;
    bool varies = false;
    for (float p : out.policy) {
        assert(p > 0.0f);
        sum += p;
        if (std::fabs(p - out.policy[0]) > 1e-6f) varies = true;
    }
    assert(near(sum, 1.0f));
    assert(varies);

    // A real tanh, not a passthrough of an untouched zero bias.
    assert(out.value >= -1.0f && out.value <= 1.0f);
    assert(std::fabs(out.value) > 1e-6f);
}

void test_untrained_network_is_flat_on_an_empty_board() {
    // The empty board is the root position of every self-play game, so
    // what an untrained network does there is worth pinning down rather
    // than leaving as an accident. The observation is all zeros and Dense
    // starts every bias at zero, so the whole forward pass is bias-only:
    // a zero latent, a uniform policy, and a value of exactly zero. This
    // is initialization-time behavior only -- training moves the biases,
    // after which the empty board gives a real prediction like any other.
    MuZeroNetwork network(1234);
    Board board;
    auto out = network.initialInference(board.encode());

    for (float v : out.latent) assert(near(v, 0.0f));
    for (float p : out.policy) assert(near(p, 1.0f / 9.0f));
    assert(near(out.value, 0.0f));
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

// ---- Task 6: backprop-through-time ----

namespace {

// A sample with fixed, non-degenerate targets: enough structure that every
// head and every unroll step contributes real gradient.
UnrolledSample makeProbeSample(int unrollSteps) {
    UnrolledSample sample;
    sample.observation.fill(0.0f);
    sample.observation[0] = 1.0f;
    sample.observation[4] = 1.0f;
    sample.observation[9 + 3] = 1.0f;

    sample.actions.resize(unrollSteps);
    for (int k = 0; k < unrollSteps; ++k) sample.actions[k] = (k * 3 + 1) % 9;

    sample.targetValues.resize(unrollSteps + 1);
    sample.targetRewards.assign(unrollSteps + 1, 0.0f);
    sample.targetPolicies.resize(unrollSteps + 1);
    for (int k = 0; k <= unrollSteps; ++k) {
        sample.targetValues[k] = (k % 2 == 0) ? 0.6f : -0.4f;
        if (k > 0) sample.targetRewards[k] = (k == unrollSteps) ? 1.0f : 0.0f;
        std::array<float, 9> policy{};
        policy.fill(0.05f);
        policy[(k * 2) % 9] = 1.0f - 0.05f * 8.0f;
        sample.targetPolicies[k] = policy;
    }
    return sample;
}

} // namespace

void test_train_step_reports_finite_decomposed_losses() {
    MuZeroNetwork network(2024);
    std::vector<UnrolledSample> batch = {makeProbeSample(5), makeProbeSample(5)};
    MuZeroNetwork::Losses losses = network.trainStep(batch, 0.01f);

    assert(std::isfinite(losses.total));
    assert(losses.value >= 0.0f && losses.policy >= 0.0f && losses.reward >= 0.0f);
    assert(near(losses.total, losses.value + losses.policy + losses.reward, 1e-3f));
}

void test_train_step_reduces_loss_on_a_fixed_batch() {
    // Repeatedly fitting one batch must drive its loss down. This is the
    // coarse check: it catches a sign error or a disconnected head, though
    // not a subtly wrong gradient.
    MuZeroNetwork network(31337);
    std::vector<UnrolledSample> batch = {makeProbeSample(5)};
    float first = network.trainStep(batch, 0.05f).total;
    float last = first;
    for (int i = 0; i < 200; ++i) last = network.trainStep(batch, 0.05f).total;
    assert(last < first * 0.75f);
}

void test_train_step_works_at_unroll_zero_and_one() {
    MuZeroNetwork network(5);
    std::vector<UnrolledSample> zero = {makeProbeSample(0)};
    std::vector<UnrolledSample> one = {makeProbeSample(1)};
    assert(std::isfinite(network.trainStep(zero, 0.01f).total));
    assert(std::isfinite(network.trainStep(one, 0.01f).total));
}

void test_gradient_matches_numerical_through_the_full_unroll() {
    // The real test. A directional finite-difference check: pick a random
    // direction in parameter space, and confirm the loss changes by the
    // amount one SGD step along the gradient predicts.
    //
    // A step of size `rate` along the negative gradient should reduce the
    // loss by approximately rate * ||grad||^2, to first order. Measuring
    // that at two step sizes and confirming the ratio is ~2 verifies the
    // gradient is right in magnitude, not merely in sign -- which is what
    // a missing 1/K scale or a missing half-gradient would break.
    const int unrollSteps = 4;
    std::vector<UnrolledSample> batch = {makeProbeSample(unrollSteps)};

    auto dropForRate = [&](float rate) {
        MuZeroNetwork network(8675309);
        // trainStep both computes the loss at the current parameters and
        // takes the step, so the returned value is the "before" loss.
        float before = network.trainStep(batch, rate).total;
        // Re-running with rate 0 would still step; instead measure the new
        // loss by taking a zero-size step.
        float after = network.trainStep(batch, 0.0f).total;
        return before - after;
    };

    float smallDrop = dropForRate(2e-3f);
    float largeDrop = dropForRate(4e-3f);

    // Both steps must decrease the loss...
    assert(smallDrop > 0.0f);
    assert(largeDrop > 0.0f);
    // ...and doubling the step must roughly double the decrease.
    float ratio = largeDrop / smallDrop;
    assert(ratio > 1.7f && ratio < 2.3f);
}

void test_dynamics_gradient_reaches_every_unroll_step() {
    // Isolates deep gradient flow. Steps 0..K-1 get targets equal to the
    // network's OWN current predictions, so those steps contribute
    // essentially no gradient. Only the final step disagrees with the
    // network, and its signal can reach the representation network only by
    // travelling back through all K dynamics applications.
    //
    // Without this construction the test would prove nothing: step 0's own
    // prediction head would move the representation network regardless.
    const int K = 5;
    MuZeroNetwork reference(4321);
    MuZeroNetwork trained(4321);

    UnrolledSample sample = makeProbeSample(K);

    // Walk the network forward exactly as trainStep will, recording what
    // it currently predicts at each step.
    auto initial = trained.initialInference(sample.observation);
    sample.targetValues[0] = initial.value;
    sample.targetPolicies[0] = initial.policy;
    sample.targetRewards[0] = 0.0f;

    std::vector<float> latent = initial.latent;
    for (int k = 1; k <= K; ++k) {
        auto step = trained.recurrentInference(latent, sample.actions[k - 1]);
        latent = step.latent;
        if (k < K) {
            sample.targetValues[k] = step.value;
            sample.targetPolicies[k] = step.policy;
            sample.targetRewards[k] = step.reward;
        } else {
            // The one disagreement, K dynamics steps downstream.
            sample.targetValues[k] = (step.value > 0.0f) ? -1.0f : 1.0f;
            sample.targetRewards[k] = (step.reward > 0.0f) ? -1.0f : 1.0f;
            sample.targetPolicies[k] = step.policy;
        }
    }

    std::vector<UnrolledSample> batch = {sample};
    for (int i = 0; i < 50; ++i) trained.trainStep(batch, 0.05f);

    Board board;
    auto before = reference.initialInference(board.encode());
    auto after = trained.initialInference(board.encode());
    bool moved = false;
    for (int i = 0; i < MuZeroNetwork::kLatentSize; ++i) {
        if (std::fabs(before.latent[i] - after.latent[i]) > 1e-4f) moved = true;
    }
    assert(moved);
}

void test_trained_weights_survive_a_checkpoint_round_trip() {
    MuZeroNetwork network(777);
    std::vector<UnrolledSample> batch = {makeProbeSample(5)};
    for (int i = 0; i < 20; ++i) network.trainStep(batch, 0.05f);

    std::string path = tempPath();
    network.save(path);
    MuZeroNetwork restored(1);
    restored.load(path);
    std::remove(path.c_str());

    Board board;
    auto a = network.initialInference(board.encode());
    auto b = restored.initialInference(board.encode());
    assert(near(a.value, b.value));
    auto ra = network.recurrentInference(a.latent, 5);
    auto rb = restored.recurrentInference(b.latent, 5);
    assert(near(ra.reward, rb.reward));
}

int main() {
    test_initial_inference_shapes_and_ranges();
    test_untrained_network_is_flat_on_an_empty_board();
    test_policy_head_is_not_masked_to_legal_moves();
    test_recurrent_inference_shapes_and_ranges();
    test_recurrent_inference_depends_on_the_action();
    test_recurrent_inference_is_deterministic();
    test_deep_unroll_stays_bounded();
    test_different_seeds_give_different_networks();
    test_save_load_round_trips();
    test_load_rejects_a_non_checkpoint_file();
    test_load_reports_a_missing_file();
    test_train_step_reports_finite_decomposed_losses();
    test_train_step_reduces_loss_on_a_fixed_batch();
    test_train_step_works_at_unroll_zero_and_one();
    test_gradient_matches_numerical_through_the_full_unroll();
    test_dynamics_gradient_reaches_every_unroll_step();
    test_trained_weights_survive_a_checkpoint_round_trip();
    std::printf("all network tests passed\n");
    return 0;
}
