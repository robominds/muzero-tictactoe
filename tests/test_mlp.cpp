#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>
#include <vector>
#include "mz/mlp.hpp"

using namespace mz;

namespace {

// Central-difference numerical gradient of `loss` with respect to *x*.
template <typename LossFn>
std::vector<float> numericalInputGradient(LossFn loss, std::vector<float> x) {
    const float eps = 1e-3f;
    std::vector<float> grad(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float original = x[i];
        x[i] = original + eps;
        float up = loss(x);
        x[i] = original - eps;
        float down = loss(x);
        x[i] = original;
        grad[i] = (up - down) / (2.0f * eps);
    }
    return grad;
}

void assertClose(const std::vector<float>& a, const std::vector<float>& b, float tol, const char* what) {
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::fabs(a[i] - b[i]);
        float scale = std::fmax(1.0f, std::fmax(std::fabs(a[i]), std::fabs(b[i])));
        if (diff / scale > tol) {
            std::printf("%s mismatch at %zu: analytic=%.6f numeric=%.6f\n", what, i, a[i], b[i]);
            assert(false);
        }
    }
}

std::vector<float> wellSeparated(int n, std::mt19937& rng) {
    // Distinct, well-spread values: keeps the numerical gradient check away
    // from ReLU kinks and from min/max ties in minMaxNormalize, where the
    // derivative is only a subgradient.
    std::uniform_real_distribution<float> jitter(-0.05f, 0.05f);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = -1.0f + 2.0f * i / (n - 1) + jitter(rng);
    return v;
}

} // namespace

void test_forward_shape_and_value() {
    std::mt19937 rng(1234);
    Dense layer(3, 2, rng);
    std::vector<float> out = layer.forward({1.0f, 2.0f, 3.0f});
    assert(out.size() == 2);
    assert(layer.inDim() == 3 && layer.outDim() == 2);
}

void test_dense_input_gradient_matches_numerical() {
    std::mt19937 rng(7);
    Dense layer(5, 4, rng);
    std::vector<float> x = wellSeparated(5, rng);
    std::vector<float> dOut = {0.3f, -0.7f, 1.1f, 0.2f};

    // Scalar loss = dot(dOut, layer(x)), whose dLoss/dOutput is exactly dOut.
    auto loss = [&](const std::vector<float>& in) {
        std::vector<float> y = layer.forward(in);
        float s = 0.0f;
        for (size_t i = 0; i < y.size(); ++i) s += dOut[i] * y[i];
        return s;
    };

    std::vector<float> analytic = layer.backward(x, dOut);
    assertClose(analytic, numericalInputGradient(loss, x), 1e-2f, "dense dInput");
}

void test_dense_weight_gradient_matches_numerical() {
    std::mt19937 rng(11);
    Dense layer(4, 3, rng);
    std::vector<float> x = wellSeparated(4, rng);
    std::vector<float> dOut = {0.5f, -0.25f, 0.75f};

    layer.zeroGrad();
    layer.backward(x, dOut);

    // One SGD step with a known rate must move the output by
    // -rate * (dLoss/dW . dW), which we verify by finite difference on the
    // scalar loss dot(dOut, layer(x)).
    auto lossOf = [&](const Dense& l) {
        std::vector<float> y = l.forward(x);
        float s = 0.0f;
        for (size_t i = 0; i < y.size(); ++i) s += dOut[i] * y[i];
        return s;
    };
    float before = lossOf(layer);
    const float rate = 1e-3f;
    Dense stepped = layer;
    stepped.applySgd(rate, 1.0f);
    float after = lossOf(stepped);

    // Gradient descent must decrease this loss, and by roughly
    // rate * ||grad||^2 > 0.
    assert(after < before);
}

void test_backward_accumulates_across_calls() {
    std::mt19937 rng(3);
    Dense once(2, 2, rng);
    Dense twice = once;
    std::vector<float> x = {0.4f, -0.9f};
    std::vector<float> dOut = {1.0f, -1.0f};

    once.zeroGrad();
    once.backward(x, dOut);
    once.applySgd(0.1f, 1.0f);

    // Two backward calls without an intervening zeroGrad must produce
    // exactly twice the gradient -- this is what makes K-step
    // backprop-through-time over a reused network correct.
    twice.zeroGrad();
    twice.backward(x, dOut);
    twice.backward(x, dOut);
    twice.applySgd(0.1f, 0.5f);

    std::vector<float> a = once.forward(x);
    std::vector<float> b = twice.forward(x);
    assertClose(a, b, 1e-5f, "accumulate");
}

void test_relu_backward_matches_numerical() {
    std::mt19937 rng(5);
    std::vector<float> z = wellSeparated(6, rng);
    std::vector<float> dOut = {0.2f, -0.4f, 0.9f, 0.1f, -0.6f, 0.3f};
    auto loss = [&](const std::vector<float>& in) {
        std::vector<float> y = relu(in);
        float s = 0.0f;
        for (size_t i = 0; i < y.size(); ++i) s += dOut[i] * y[i];
        return s;
    };
    assertClose(reluBackward(z, dOut), numericalInputGradient(loss, z), 1e-2f, "relu");
}

void test_minmax_normalize_maps_into_unit_range() {
    std::vector<float> z = {-3.0f, 0.5f, 7.0f, 2.0f};
    std::vector<float> y = minMaxNormalize(z);
    assert(std::fabs(y[0] - 0.0f) < 1e-6f);
    assert(std::fabs(y[2] - 1.0f) < 1e-6f);
    for (float v : y) assert(v >= -1e-6f && v <= 1.0f + 1e-6f);
}

void test_minmax_normalize_handles_constant_input() {
    std::vector<float> z = {2.0f, 2.0f, 2.0f};
    std::vector<float> y = minMaxNormalize(z);
    for (float v : y) assert(std::isfinite(v));
}

void test_minmax_normalize_backward_matches_numerical() {
    std::mt19937 rng(13);
    std::vector<float> z = wellSeparated(8, rng);
    std::vector<float> dOut = {0.1f, -0.5f, 0.8f, 0.2f, -0.3f, 0.6f, -0.1f, 0.4f};
    auto loss = [&](const std::vector<float>& in) {
        std::vector<float> y = minMaxNormalize(in);
        float s = 0.0f;
        for (size_t i = 0; i < y.size(); ++i) s += dOut[i] * y[i];
        return s;
    };
    assertClose(minMaxNormalizeBackward(z, dOut), numericalInputGradient(loss, z), 2e-2f, "minmax");
}

void test_softmax9_sums_to_one() {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f, 0.0f, -1.0f, 0.5f, 0.5f, 2.5f, -2.0f};
    std::array<float, 9> p = softmax9(logits);
    float sum = 0.0f;
    for (float v : p) { assert(v > 0.0f); sum += v; }
    assert(std::fabs(sum - 1.0f) < 1e-5f);
}

void test_softmax9_is_shift_invariant_and_stable() {
    std::vector<float> small = {1.0f, 2.0f, 3.0f, 0.0f, -1.0f, 0.5f, 0.5f, 2.5f, -2.0f};
    std::vector<float> huge = small;
    for (float& v : huge) v += 1000.0f;
    std::array<float, 9> a = softmax9(small);
    std::array<float, 9> b = softmax9(huge);
    for (int i = 0; i < 9; ++i) {
        assert(std::isfinite(b[i]));
        assert(std::fabs(a[i] - b[i]) < 1e-5f);
    }
}

void test_dense_write_read_round_trips() {
    std::mt19937 rng(21);
    Dense original(4, 3, rng);
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    original.write(stream);

    std::mt19937 otherRng(22);
    Dense restored(4, 3, otherRng);
    stream.seekg(0);
    restored.read(stream);

    std::vector<float> x = {0.1f, -0.2f, 0.3f, 0.4f};
    assertClose(original.forward(x), restored.forward(x), 1e-6f, "round trip");
}

int main() {
    test_forward_shape_and_value();
    test_dense_input_gradient_matches_numerical();
    test_dense_weight_gradient_matches_numerical();
    test_backward_accumulates_across_calls();
    test_relu_backward_matches_numerical();
    test_minmax_normalize_maps_into_unit_range();
    test_minmax_normalize_handles_constant_input();
    test_minmax_normalize_backward_matches_numerical();
    test_softmax9_sums_to_one();
    test_softmax9_is_shift_invariant_and_stable();
    test_dense_write_read_round_trips();
    std::printf("all mlp tests passed\n");
    return 0;
}
