#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "mz/mlp.hpp"
#include "mz/targets.hpp"

namespace mz {

// MuZero's three functions, each a small MLP built from Dense.
//
//   representation h : observation -> latent
//   dynamics       g : (latent, action) -> (latent, reward)
//   prediction     f : latent -> (policy, value)
//
// There is deliberately NO decoder from latent back to a board. Nothing
// here can look at a latent and say which position it is. A latent only
// has to support good policy, value and reward predictions -- it is never
// asked to reconstruct the observation. That is the constraint that
// separates MuZero from AlphaZero, which always has the real board.
//
// Perspective conventions:
//   value  -- from the perspective of the player to move at that latent
//   reward -- from the perspective of the player who took the action
class MuZeroNetwork {
public:
    static constexpr int kObservationSize = 18;
    static constexpr int kActionSize = 9;
    static constexpr int kLatentSize = 32;
    static constexpr int kHiddenSize = 64;

    struct InitialInference {
        std::vector<float> latent;
        std::array<float, kActionSize> policy;
        float value;
    };

    struct RecurrentInference {
        std::vector<float> latent;
        float reward;
        std::array<float, kActionSize> policy;
        float value;
    };

    struct Losses {
        float total = 0.0f;
        float value = 0.0f;
        float policy = 0.0f;
        float reward = 0.0f;
    };

    explicit MuZeroNetwork(std::uint32_t seed);
    MuZeroNetwork();

    // h then f. Used at the search root, on the real board's encoding --
    // the only place an observation enters the system.
    InitialInference initialInference(const std::array<float, kObservationSize>& observation) const;

    // g then f. Every non-root node in the tree is reached this way, with
    // no reference to the real board at all.
    // Precondition: latent.size() == kLatentSize, 0 <= action < 9 (asserted).
    RecurrentInference recurrentInference(const std::vector<float>& latent, int action) const;

    // One SGD step over a batch of unrolled samples. Implemented in Task 6.
    Losses trainStep(const std::vector<UnrolledSample>& batch, float learningRate);

    // The eight layers, addressable for gradient checking. A hand-written
    // network with no autograd behind it has to be checkable from outside,
    // and save()/load() already round-trips exactly this state.
    enum class LayerId {
        RepresentationFc1, RepresentationFc2,
        DynamicsFc1, DynamicsFc2, DynamicsReward,
        PredictionFc1, PredictionPolicy, PredictionValue,
    };
    Dense& layer(LayerId id);
    const Dense& layer(LayerId id) const;

    // MuZero scales the gradient entering the dynamics network from its
    // latent input by 0.5 at every unroll step, so gradient magnitude does
    // not compound across the recurrence. That is a DELIBERATE deviation
    // from the true gradient of the loss -- which means a per-parameter
    // numerical gradient check, which necessarily measures the true
    // gradient, cannot agree with the reverse pass unless the deviation is
    // switched off first. Hence this seam. Training always leaves it at
    // 0.5; only gradient checks touch it.
    static constexpr float kHalfGradient = 0.5f;
    float dynamicsGradientScale() const { return dynamicsGradientScale_; }
    void setDynamicsGradientScale(float scale) { dynamicsGradientScale_ = scale; }

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    // Forward pieces shared by inference and training. Each returns the
    // pre-activations training needs; inference discards them.
    struct HiddenTrace {
        std::vector<float> preActivation;   // before ReLU
        std::vector<float> activation;      // after ReLU
    };

    HiddenTrace representationHidden(const std::array<float, kObservationSize>& observation) const;
    HiddenTrace dynamicsHidden(const std::vector<float>& dynamicsInput) const;
    HiddenTrace predictionHidden(const std::vector<float>& latent) const;

    static std::vector<float> makeDynamicsInput(const std::vector<float>& latent, int action);

    Dense hFc1_, hFc2_;                  // representation
    Dense gFc1_, gFc2_, gReward_;        // dynamics
    Dense fFc1_, fPolicy_, fValue_;      // prediction

    float dynamicsGradientScale_ = kHalfGradient;
};

} // namespace mz
