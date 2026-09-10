#pragma once
#include <array>
#include <iosfwd>
#include <random>
#include <vector>

namespace mz {

// A fully-connected layer y = Wx + b, with hand-written forward and
// backward passes. This is the only differentiation primitive in the
// project -- the three MuZero networks are built entirely from it.
//
// Two properties are load-bearing for MuZero specifically:
//
//   backward() takes the input EXPLICITLY rather than caching it during
//   forward(). The dynamics network is applied K times inside one loss,
//   each time on a different latent, and backprop-through-time needs each
//   of those inputs again on the way back. A single cached input would
//   silently use the wrong one.
//
//   backward() ACCUMULATES into the gradient buffers instead of assigning.
//   Those same K applications must sum into one shared gradient. Call
//   zeroGrad() once per training step -- never once per unroll step.
class Dense {
public:
    Dense() = default;
    Dense(int inDim, int outDim, std::mt19937& rng);

    int inDim() const { return inDim_; }
    int outDim() const { return outDim_; }

    // Wx + b. const, so inference paths can hold the network by const ref.
    std::vector<float> forward(const std::vector<float>& input) const;

    // Given dLoss/dOutput, accumulates dLoss/dW and dLoss/db and returns
    // dLoss/dInput. `input` must be the input this gradient came from.
    std::vector<float> backward(const std::vector<float>& input,
                                const std::vector<float>& dOutput);

    void zeroGrad();
    // theta -= learningRate * scale * grad. `scale` is normally 1/batchSize.
    void applySgd(float learningRate, float scale);

    // Parameter access. Public because a hand-written layer with no
    // autograd behind it has to be gradient-checkable from outside, and
    // because write()/read() already exposes exactly this state.
    // Precondition on all four: indices in range (asserted).
    float weightAt(int outIndex, int inIndex) const;
    void setWeightAt(int outIndex, int inIndex, float value);
    float weightGradientAt(int outIndex, int inIndex) const;
    float biasGradientAt(int outIndex) const;

    void write(std::ostream& out) const;
    void read(std::istream& in);

private:
    int inDim_ = 0;
    int outDim_ = 0;
    std::vector<float> w_;      // outDim_ * inDim_, row-major
    std::vector<float> b_;      // outDim_
    std::vector<float> gradW_;
    std::vector<float> gradB_;
};

std::vector<float> relu(const std::vector<float>& z);
// `z` is the PRE-activation; ReLU passes gradient only where z > 0.
std::vector<float> reluBackward(const std::vector<float>& z, const std::vector<float>& dOut);

// Numerically stable softmax over exactly 9 logits (the action space).
std::array<float, 9> softmax9(const std::vector<float>& logits);

// Scales z into [0, 1] across its own elements, per MuZero appendix G:
//   y_i = (z_i - min(z)) / max(max(z) - min(z), 1e-5)
// Applied to every latent produced by representation and dynamics. Without
// it, applying dynamics repeatedly lets latent magnitudes drift and the
// recurrence destabilizes.
std::vector<float> minMaxNormalize(const std::vector<float>& z);

// The gradient of minMaxNormalize, including the terms that flow through
// min(z) and max(z) themselves -- treating them as constants would pass a
// casual eyeball but fails the numerical gradient check in test_mlp.
std::vector<float> minMaxNormalizeBackward(const std::vector<float>& z, const std::vector<float>& dOut);

} // namespace mz
