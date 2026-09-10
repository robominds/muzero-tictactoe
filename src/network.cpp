#include "mz/network.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

namespace mz {

namespace {
constexpr char kCheckpointMagic[4] = {'M', 'Z', 'N', 'N'};
constexpr std::uint32_t kCheckpointVersion = 1;
} // namespace

MuZeroNetwork::MuZeroNetwork(std::uint32_t seed) {
    std::mt19937 rng(seed);
    hFc1_ = Dense(kObservationSize, kHiddenSize, rng);
    hFc2_ = Dense(kHiddenSize, kLatentSize, rng);
    gFc1_ = Dense(kLatentSize + kActionSize, kHiddenSize, rng);
    gFc2_ = Dense(kHiddenSize, kLatentSize, rng);
    gReward_ = Dense(kHiddenSize, 1, rng);
    fFc1_ = Dense(kLatentSize, kHiddenSize, rng);
    fPolicy_ = Dense(kHiddenSize, kActionSize, rng);
    fValue_ = Dense(kHiddenSize, 1, rng);
}

MuZeroNetwork::MuZeroNetwork() : MuZeroNetwork(std::random_device{}()) {}

std::vector<float> MuZeroNetwork::makeDynamicsInput(const std::vector<float>& latent, int action) {
    assert(static_cast<int>(latent.size()) == kLatentSize);
    assert(action >= 0 && action < kActionSize);
    // Latent with the action appended as a one-hot. This is how an action
    // enters the model at all -- there is no board to apply it to.
    std::vector<float> input(kLatentSize + kActionSize, 0.0f);
    std::copy(latent.begin(), latent.end(), input.begin());
    input[kLatentSize + action] = 1.0f;
    return input;
}

MuZeroNetwork::HiddenTrace MuZeroNetwork::representationHidden(
    const std::array<float, kObservationSize>& observation) const {
    std::vector<float> input(observation.begin(), observation.end());
    HiddenTrace trace;
    trace.preActivation = hFc1_.forward(input);
    trace.activation = relu(trace.preActivation);
    return trace;
}

MuZeroNetwork::HiddenTrace MuZeroNetwork::dynamicsHidden(const std::vector<float>& dynamicsInput) const {
    HiddenTrace trace;
    trace.preActivation = gFc1_.forward(dynamicsInput);
    trace.activation = relu(trace.preActivation);
    return trace;
}

MuZeroNetwork::HiddenTrace MuZeroNetwork::predictionHidden(const std::vector<float>& latent) const {
    HiddenTrace trace;
    trace.preActivation = fFc1_.forward(latent);
    trace.activation = relu(trace.preActivation);
    return trace;
}

MuZeroNetwork::InitialInference MuZeroNetwork::initialInference(
    const std::array<float, kObservationSize>& observation) const {
    HiddenTrace hidden = representationHidden(observation);
    std::vector<float> latent = minMaxNormalize(hFc2_.forward(hidden.activation));

    HiddenTrace predictionTrace = predictionHidden(latent);
    InitialInference out;
    out.latent = std::move(latent);
    out.policy = softmax9(fPolicy_.forward(predictionTrace.activation));
    out.value = std::tanh(fValue_.forward(predictionTrace.activation)[0]);
    return out;
}

MuZeroNetwork::RecurrentInference MuZeroNetwork::recurrentInference(const std::vector<float>& latent,
                                                                    int action) const {
    std::vector<float> input = makeDynamicsInput(latent, action);
    HiddenTrace hidden = dynamicsHidden(input);

    std::vector<float> nextLatent = minMaxNormalize(gFc2_.forward(hidden.activation));
    float reward = std::tanh(gReward_.forward(hidden.activation)[0]);

    HiddenTrace predictionTrace = predictionHidden(nextLatent);
    RecurrentInference out;
    out.latent = std::move(nextLatent);
    out.reward = reward;
    out.policy = softmax9(fPolicy_.forward(predictionTrace.activation));
    out.value = std::tanh(fValue_.forward(predictionTrace.activation)[0]);
    return out;
}

void MuZeroNetwork::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("MuZeroNetwork::save: cannot open " + path);

    out.write(kCheckpointMagic, sizeof(kCheckpointMagic));
    std::uint32_t header[5] = {kCheckpointVersion,
                               static_cast<std::uint32_t>(kObservationSize),
                               static_cast<std::uint32_t>(kActionSize),
                               static_cast<std::uint32_t>(kLatentSize),
                               static_cast<std::uint32_t>(kHiddenSize)};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));

    for (const Dense* layer : {&hFc1_, &hFc2_, &gFc1_, &gFc2_, &gReward_, &fFc1_, &fPolicy_, &fValue_}) {
        layer->write(out);
    }
    if (!out) throw std::runtime_error("MuZeroNetwork::save: write failed for " + path);
}

void MuZeroNetwork::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("MuZeroNetwork::load: cannot open " + path);

    char magic[4] = {};
    std::uint32_t header[5] = {};
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(header), sizeof(header));

    bool headerOk = in && std::equal(std::begin(magic), std::end(magic), std::begin(kCheckpointMagic)) &&
                    header[0] == kCheckpointVersion &&
                    header[1] == static_cast<std::uint32_t>(kObservationSize) &&
                    header[2] == static_cast<std::uint32_t>(kActionSize) &&
                    header[3] == static_cast<std::uint32_t>(kLatentSize) &&
                    header[4] == static_cast<std::uint32_t>(kHiddenSize);
    if (!headerOk) {
        throw std::runtime_error(
            "MuZeroNetwork::load: not a valid checkpoint (bad magic/version/shape): " + path);
    }

    for (Dense* layer : {&hFc1_, &hFc2_, &gFc1_, &gFc2_, &gReward_, &fFc1_, &fPolicy_, &fValue_}) {
        layer->read(in);
    }
    if (!in) throw std::runtime_error("MuZeroNetwork::load: truncated file " + path);
}

const Dense& MuZeroNetwork::layer(LayerId id) const {
    switch (id) {
        case LayerId::RepresentationFc1: return hFc1_;
        case LayerId::RepresentationFc2: return hFc2_;
        case LayerId::DynamicsFc1:       return gFc1_;
        case LayerId::DynamicsFc2:       return gFc2_;
        case LayerId::DynamicsReward:    return gReward_;
        case LayerId::PredictionFc1:     return fFc1_;
        case LayerId::PredictionPolicy:  return fPolicy_;
        case LayerId::PredictionValue:   return fValue_;
    }
    // No default case above, so -Wswitch flags any enumerator added later.
    assert(false && "MuZeroNetwork::layer: unhandled LayerId");
    return hFc1_;
}

Dense& MuZeroNetwork::layer(LayerId id) {
    // Delegates to the const overload rather than repeating the switch.
    return const_cast<Dense&>(static_cast<const MuZeroNetwork*>(this)->layer(id));
}

MuZeroNetwork::Losses MuZeroNetwork::trainStep(const std::vector<UnrolledSample>& batch,
                                               float learningRate) {
    assert(!batch.empty());

    for (Dense* layer : {&hFc1_, &hFc2_, &gFc1_, &gFc2_, &gReward_, &fFc1_, &fPolicy_, &fValue_}) {
        layer->zeroGrad();
    }

    Losses losses;

    for (const UnrolledSample& sample : batch) {
        const int K = static_cast<int>(sample.actions.size());
        assert(static_cast<int>(sample.targetValues.size()) == K + 1);
        assert(static_cast<int>(sample.targetRewards.size()) == K + 1);
        assert(static_cast<int>(sample.targetPolicies.size()) == K + 1);

        // Every unroll step past 0 contributes at 1/K, so a deeply
        // unrolled sample does not outweigh a shallow one.
        const float tailScale = (K > 0) ? 1.0f / static_cast<float>(K) : 1.0f;
        auto lossScale = [&](int k) { return k == 0 ? 1.0f : tailScale; };

        // ---- forward, keeping everything the backward pass will need ----

        HiddenTrace hTrace = representationHidden(sample.observation);
        std::vector<std::vector<float>> latentPre(K + 1);   // gFc2/hFc2 output, pre-normalization
        std::vector<std::vector<float>> latent(K + 1);      // after minMaxNormalize

        latentPre[0] = hFc2_.forward(hTrace.activation);
        latent[0] = minMaxNormalize(latentPre[0]);

        std::vector<std::vector<float>> dynamicsInput(K);
        std::vector<HiddenTrace> dynamicsTrace(K);
        std::vector<float> reward(K + 1, 0.0f);
        std::vector<float> rewardPre(K + 1, 0.0f);

        for (int k = 0; k < K; ++k) {
            dynamicsInput[k] = makeDynamicsInput(latent[k], sample.actions[k]);
            dynamicsTrace[k] = dynamicsHidden(dynamicsInput[k]);
            latentPre[k + 1] = gFc2_.forward(dynamicsTrace[k].activation);
            latent[k + 1] = minMaxNormalize(latentPre[k + 1]);
            rewardPre[k + 1] = gReward_.forward(dynamicsTrace[k].activation)[0];
            reward[k + 1] = std::tanh(rewardPre[k + 1]);
        }

        std::vector<HiddenTrace> predictionTrace(K + 1);
        std::vector<std::array<float, kActionSize>> policy(K + 1);
        std::vector<float> value(K + 1, 0.0f);

        for (int k = 0; k <= K; ++k) {
            predictionTrace[k] = predictionHidden(latent[k]);
            policy[k] = softmax9(fPolicy_.forward(predictionTrace[k].activation));
            value[k] = std::tanh(fValue_.forward(predictionTrace[k].activation)[0]);
        }

        // ---- loss ----

        for (int k = 0; k <= K; ++k) {
            float scale = lossScale(k);
            float valueError = value[k] - sample.targetValues[k];
            losses.value += scale * valueError * valueError;
            for (int a = 0; a < kActionSize; ++a) {
                losses.policy -= scale * sample.targetPolicies[k][a] * std::log(policy[k][a] + 1e-8f);
            }
            // targetRewards[0] is always 0 and excluded: step 0 has no
            // incoming transition to predict a reward for.
            if (k > 0) {
                float rewardError = reward[k] - sample.targetRewards[k];
                losses.reward += scale * rewardError * rewardError;
            }
        }

        // ---- backward through time ----

        // dLatent[k] accumulates dLoss/dLatent[k] from two sources: the
        // prediction head at step k, and the dynamics step k -> k+1. The
        // single reverse loop guarantees both have arrived before it is
        // used.
        std::vector<std::vector<float>> dLatent(K + 1, std::vector<float>(kLatentSize, 0.0f));

        for (int k = K; k >= 0; --k) {
            float scale = lossScale(k);

            // prediction head at step k
            float dValuePre = scale * 2.0f * (value[k] - sample.targetValues[k]) *
                              (1.0f - value[k] * value[k]);
            std::vector<float> dPolicyLogits(kActionSize);
            for (int a = 0; a < kActionSize; ++a) {
                // d(cross-entropy o softmax)/d(logit) = p - target
                dPolicyLogits[a] = scale * (policy[k][a] - sample.targetPolicies[k][a]);
            }

            std::vector<float> dPredictionHidden =
                fValue_.backward(predictionTrace[k].activation, {dValuePre});
            std::vector<float> dFromPolicy =
                fPolicy_.backward(predictionTrace[k].activation, dPolicyLogits);
            for (int i = 0; i < kHiddenSize; ++i) dPredictionHidden[i] += dFromPolicy[i];

            std::vector<float> dPredictionPre =
                reluBackward(predictionTrace[k].preActivation, dPredictionHidden);
            std::vector<float> dFromPrediction = fFc1_.backward(latent[k], dPredictionPre);
            for (int i = 0; i < kLatentSize; ++i) dLatent[k][i] += dFromPrediction[i];

            if (k > 0) {
                // dynamics step k-1 -> k
                std::vector<float> dLatentPre = minMaxNormalizeBackward(latentPre[k], dLatent[k]);
                std::vector<float> dDynamicsHidden =
                    gFc2_.backward(dynamicsTrace[k - 1].activation, dLatentPre);

                float dRewardPre = scale * 2.0f * (reward[k] - sample.targetRewards[k]) *
                                   (1.0f - reward[k] * reward[k]);
                std::vector<float> dFromReward =
                    gReward_.backward(dynamicsTrace[k - 1].activation, {dRewardPre});
                for (int i = 0; i < kHiddenSize; ++i) dDynamicsHidden[i] += dFromReward[i];

                std::vector<float> dDynamicsPre =
                    reluBackward(dynamicsTrace[k - 1].preActivation, dDynamicsHidden);
                std::vector<float> dDynamicsInput =
                    gFc1_.backward(dynamicsInput[k - 1], dDynamicsPre);

                // The half gradient. Scaling what flows back into the
                // dynamics input by 0.5 at every step keeps gradient
                // magnitude from compounding across the recurrence. One
                // line, easy to omit, and omitting it destabilizes latents
                // as the unroll deepens. Always 0.5 in training; a
                // gradient check sets it to 1 to recover the true
                // gradient. See MuZeroNetwork::setDynamicsGradientScale.
                for (int i = 0; i < kLatentSize; ++i) {
                    dLatent[k - 1][i] += dynamicsGradientScale_ * dDynamicsInput[i];
                }
            } else {
                // representation network
                std::vector<float> dLatentPre = minMaxNormalizeBackward(latentPre[0], dLatent[0]);
                std::vector<float> dHidden = hFc2_.backward(hTrace.activation, dLatentPre);
                std::vector<float> dHiddenPre = reluBackward(hTrace.preActivation, dHidden);
                std::vector<float> observation(sample.observation.begin(), sample.observation.end());
                hFc1_.backward(observation, dHiddenPre);
            }
        }
    }

    const float scale = 1.0f / static_cast<float>(batch.size());
    for (Dense* layer : {&hFc1_, &hFc2_, &gFc1_, &gFc2_, &gReward_, &fFc1_, &fPolicy_, &fValue_}) {
        layer->applySgd(learningRate, scale);
    }

    losses.value *= scale;
    losses.policy *= scale;
    losses.reward *= scale;
    losses.total = losses.value + losses.policy + losses.reward;
    return losses;
}

} // namespace mz
