#include "mz/mlp.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <istream>
#include <ostream>

namespace mz {

namespace {
constexpr float kMinMaxFloor = 1e-5f;
}

Dense::Dense(int inDim, int outDim, std::mt19937& rng)
    : inDim_(inDim), outDim_(outDim),
      w_(static_cast<size_t>(inDim) * outDim), b_(outDim, 0.0f),
      gradW_(static_cast<size_t>(inDim) * outDim, 0.0f), gradB_(outDim, 0.0f) {
    // Scaled by fan-in, since the three networks here have quite different
    // input widths (18, 41, 32) and a single fixed range would leave some
    // of them saturated and others barely moving.
    float limit = 1.0f / std::sqrt(static_cast<float>(inDim));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (float& v : w_) v = dist(rng);
}

std::vector<float> Dense::forward(const std::vector<float>& input) const {
    assert(static_cast<int>(input.size()) == inDim_);
    std::vector<float> out(outDim_);
    for (int o = 0; o < outDim_; ++o) {
        float sum = b_[o];
        const float* row = &w_[static_cast<size_t>(o) * inDim_];
        for (int i = 0; i < inDim_; ++i) sum += row[i] * input[i];
        out[o] = sum;
    }
    return out;
}

std::vector<float> Dense::backward(const std::vector<float>& input,
                                   const std::vector<float>& dOutput) {
    assert(static_cast<int>(input.size()) == inDim_);
    assert(static_cast<int>(dOutput.size()) == outDim_);
    std::vector<float> dInput(inDim_, 0.0f);
    for (int o = 0; o < outDim_; ++o) {
        float d = dOutput[o];
        float* gradRow = &gradW_[static_cast<size_t>(o) * inDim_];
        const float* row = &w_[static_cast<size_t>(o) * inDim_];
        for (int i = 0; i < inDim_; ++i) {
            gradRow[i] += d * input[i];   // accumulate, never assign
            dInput[i] += d * row[i];
        }
        gradB_[o] += d;
    }
    return dInput;
}

void Dense::zeroGrad() {
    std::fill(gradW_.begin(), gradW_.end(), 0.0f);
    std::fill(gradB_.begin(), gradB_.end(), 0.0f);
}

void Dense::applySgd(float learningRate, float scale) {
    float step = learningRate * scale;
    for (size_t i = 0; i < w_.size(); ++i) w_[i] -= step * gradW_[i];
    for (size_t i = 0; i < b_.size(); ++i) b_[i] -= step * gradB_[i];
}

float Dense::weightAt(int outIndex, int inIndex) const {
    assert(outIndex >= 0 && outIndex < outDim_);
    assert(inIndex >= 0 && inIndex < inDim_);
    return w_[static_cast<size_t>(outIndex) * inDim_ + inIndex];
}

void Dense::setWeightAt(int outIndex, int inIndex, float value) {
    assert(outIndex >= 0 && outIndex < outDim_);
    assert(inIndex >= 0 && inIndex < inDim_);
    w_[static_cast<size_t>(outIndex) * inDim_ + inIndex] = value;
}

float Dense::weightGradientAt(int outIndex, int inIndex) const {
    assert(outIndex >= 0 && outIndex < outDim_);
    assert(inIndex >= 0 && inIndex < inDim_);
    return gradW_[static_cast<size_t>(outIndex) * inDim_ + inIndex];
}

float Dense::biasGradientAt(int outIndex) const {
    assert(outIndex >= 0 && outIndex < outDim_);
    return gradB_[outIndex];
}

void Dense::write(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(w_.data()), w_.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(b_.data()), b_.size() * sizeof(float));
}

void Dense::read(std::istream& in) {
    in.read(reinterpret_cast<char*>(w_.data()), w_.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(b_.data()), b_.size() * sizeof(float));
}

std::vector<float> relu(const std::vector<float>& z) {
    std::vector<float> out(z.size());
    for (size_t i = 0; i < z.size(); ++i) out[i] = z[i] > 0.0f ? z[i] : 0.0f;
    return out;
}

std::vector<float> reluBackward(const std::vector<float>& z, const std::vector<float>& dOut) {
    assert(z.size() == dOut.size());
    std::vector<float> out(z.size());
    for (size_t i = 0; i < z.size(); ++i) out[i] = z[i] > 0.0f ? dOut[i] : 0.0f;
    return out;
}

std::array<float, 9> softmax9(const std::vector<float>& logits) {
    assert(logits.size() == 9);
    float maxLogit = logits[0];
    for (float l : logits) maxLogit = std::max(maxLogit, l);
    std::array<float, 9> out{};
    float sum = 0.0f;
    for (int i = 0; i < 9; ++i) {
        out[i] = std::exp(logits[i] - maxLogit);
        sum += out[i];
    }
    for (float& v : out) v /= sum;
    return out;
}

std::vector<float> minMaxNormalize(const std::vector<float>& z) {
    assert(!z.empty());
    float lo = *std::min_element(z.begin(), z.end());
    float hi = *std::max_element(z.begin(), z.end());
    float denom = std::max(hi - lo, kMinMaxFloor);
    std::vector<float> out(z.size());
    for (size_t i = 0; i < z.size(); ++i) out[i] = (z[i] - lo) / denom;
    return out;
}

std::vector<float> minMaxNormalizeBackward(const std::vector<float>& z, const std::vector<float>& dOut) {
    assert(z.size() == dOut.size());
    // With y_i = (z_i - m) / d where m = min(z), M = max(z), d = M - m:
    //   dy_i/dz_j = (delta_ij - [j==argmin]) / d
    //               - y_i / d * ([j==argmax] - [j==argmin])
    // Summing against dOut and writing S = sum(dOut), T = sum(dOut_i * y_i):
    //   dL/dz_j = dOut_j/d - [j==argmin]*S/d - ([j==argmax]-[j==argmin])*T/d
    size_t argmin = std::min_element(z.begin(), z.end()) - z.begin();
    size_t argmax = std::max_element(z.begin(), z.end()) - z.begin();
    float lo = z[argmin], hi = z[argmax];
    float range = hi - lo;
    float denom = std::max(range, kMinMaxFloor);

    std::vector<float> out(z.size());
    float S = 0.0f, T = 0.0f;
    for (size_t i = 0; i < z.size(); ++i) {
        S += dOut[i];
        T += dOut[i] * ((z[i] - lo) / denom);
    }
    // When the range is clamped by the floor, the denominator is a constant
    // and its derivative term drops out.
    bool clamped = range < kMinMaxFloor;
    for (size_t j = 0; j < z.size(); ++j) {
        float g = dOut[j] / denom;
        if (j == argmin) g -= S / denom;
        if (!clamped) {
            float indicator = (j == argmax ? 1.0f : 0.0f) - (j == argmin ? 1.0f : 0.0f);
            g -= indicator * T / denom;
        }
        out[j] = g;
    }
    return out;
}

} // namespace mz
