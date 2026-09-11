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

void MuZeroNetwork::fillDynamicsInput(const std::vector<float>& latent, int action,
                                      std::vector<float>& out) {
    assert(static_cast<int>(latent.size()) == kLatentSize);
    assert(action >= 0 && action < kActionSize);
    // assign, not resize: the one-hot tail must be zeroed even when this
    // buffer is being reused from a previous action.
    out.assign(kLatentSize + kActionSize, 0.0f);
    std::copy(latent.begin(), latent.end(), out.begin());
    out[kLatentSize + action] = 1.0f;
}





MuZeroNetwork::InitialInference MuZeroNetwork::initialInference(
    const std::array<float, kObservationSize>& observation) const {
    Workspace workspace;
    return initialInference(observation, workspace);
}

MuZeroNetwork::InitialInference MuZeroNetwork::initialInference(
    const std::array<float, kObservationSize>& observation, Workspace& ws) const {
    ws.observation.assign(observation.begin(), observation.end());
    hFc1_.forwardInto(ws.observation, ws.trunkPre);
    reluInto(ws.trunkPre, ws.trunk);
    hFc2_.forwardInto(ws.trunk, ws.latentPre);

    InitialInference out;
    // The latent is real output, not scratch -- the caller keeps it.
    minMaxNormalizeInto(ws.latentPre, out.latent);

    fFc1_.forwardInto(out.latent, ws.predPre);
    reluInto(ws.predPre, ws.pred);
    fPolicy_.forwardInto(ws.pred, ws.logits);
    out.policy = softmax9(ws.logits);
    fValue_.forwardInto(ws.pred, ws.scalar);
    out.value = std::tanh(ws.scalar[0]);
    return out;
}

MuZeroNetwork::RecurrentInference MuZeroNetwork::recurrentInference(const std::vector<float>& latent,
                                                                    int action) const {
    Workspace workspace;
    return recurrentInference(latent, action, workspace);
}

MuZeroNetwork::RecurrentInference MuZeroNetwork::recurrentInference(const std::vector<float>& latent,
                                                                    int action, Workspace& ws) const {
    fillDynamicsInput(latent, action, ws.dynamicsInput);
    gFc1_.forwardInto(ws.dynamicsInput, ws.trunkPre);
    reluInto(ws.trunkPre, ws.trunk);
    gFc2_.forwardInto(ws.trunk, ws.latentPre);

    RecurrentInference out;
    minMaxNormalizeInto(ws.latentPre, out.latent);
    gReward_.forwardInto(ws.trunk, ws.scalar);
    out.reward = std::tanh(ws.scalar[0]);

    fFc1_.forwardInto(out.latent, ws.predPre);
    reluInto(ws.predPre, ws.pred);
    fPolicy_.forwardInto(ws.pred, ws.logits);
    out.policy = softmax9(ws.logits);
    fValue_.forwardInto(ws.pred, ws.scalar);
    out.value = std::tanh(ws.scalar[0]);
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
    TrainScratch& ws = scratch_;

    for (const UnrolledSample& sample : batch) {
        const int K = static_cast<int>(sample.actions.size());
        assert(static_cast<int>(sample.targetValues.size()) == K + 1);
        assert(static_cast<int>(sample.targetRewards.size()) == K + 1);
        assert(static_cast<int>(sample.targetPolicies.size()) == K + 1);

        // Every unroll step past 0 contributes at 1/K, so a deeply
        // unrolled sample does not outweigh a shallow one.
        const float tailScale = (K > 0) ? 1.0f / static_cast<float>(K) : 1.0f;
        auto lossScale = [&](int k) { return k == 0 ? 1.0f : tailScale; };

        // These resizes are no-ops after the first sample of the first
        // batch: K is fixed and the inner buffers keep their capacity.
        ws.latentPre.resize(K + 1);
        ws.latent.resize(K + 1);
        ws.dynamicsInput.resize(K);
        ws.dynamicsPre.resize(K);
        ws.dynamicsAct.resize(K);
        ws.predictionPre.resize(K + 1);
        ws.predictionAct.resize(K + 1);
        ws.policy.resize(K + 1);
        ws.reward.assign(K + 1, 0.0f);
        ws.value.assign(K + 1, 0.0f);
        ws.dLatent.resize(K + 1);

        // ---- forward, keeping everything the backward pass will need ----

        ws.observation.assign(sample.observation.begin(), sample.observation.end());
        hFc1_.forwardInto(ws.observation, ws.hiddenPre);
        reluInto(ws.hiddenPre, ws.hidden);
        hFc2_.forwardInto(ws.hidden, ws.latentPre[0]);
        minMaxNormalizeInto(ws.latentPre[0], ws.latent[0]);

        for (int k = 0; k < K; ++k) {
            fillDynamicsInput(ws.latent[k], sample.actions[k], ws.dynamicsInput[k]);
            gFc1_.forwardInto(ws.dynamicsInput[k], ws.dynamicsPre[k]);
            reluInto(ws.dynamicsPre[k], ws.dynamicsAct[k]);
            gFc2_.forwardInto(ws.dynamicsAct[k], ws.latentPre[k + 1]);
            minMaxNormalizeInto(ws.latentPre[k + 1], ws.latent[k + 1]);
            gReward_.forwardInto(ws.dynamicsAct[k], ws.scalarGrad);
            ws.reward[k + 1] = std::tanh(ws.scalarGrad[0]);
        }

        for (int k = 0; k <= K; ++k) {
            fFc1_.forwardInto(ws.latent[k], ws.predictionPre[k]);
            reluInto(ws.predictionPre[k], ws.predictionAct[k]);
            fPolicy_.forwardInto(ws.predictionAct[k], ws.dPolicyLogits);
            ws.policy[k] = softmax9(ws.dPolicyLogits);
            fValue_.forwardInto(ws.predictionAct[k], ws.scalarGrad);
            ws.value[k] = std::tanh(ws.scalarGrad[0]);
        }

        // ---- loss ----

        for (int k = 0; k <= K; ++k) {
            float scale = lossScale(k);
            float valueError = ws.value[k] - sample.targetValues[k];
            losses.value += scale * valueError * valueError;
            for (int a = 0; a < kActionSize; ++a) {
                losses.policy -= scale * sample.targetPolicies[k][a] * std::log(ws.policy[k][a] + 1e-8f);
            }
            // targetRewards[0] is always 0 and excluded: step 0 has no
            // incoming transition to predict a reward for.
            if (k > 0) {
                float rewardError = ws.reward[k] - sample.targetRewards[k];
                losses.reward += scale * rewardError * rewardError;
            }
        }

        // ---- backward through time ----

        // dLatent[k] accumulates dLoss/dLatent[k] from two sources: the
        // prediction head at step k, and the dynamics step k -> k+1. The
        // single reverse loop guarantees both have arrived before it is
        // used.
        for (int k = 0; k <= K; ++k) ws.dLatent[k].assign(kLatentSize, 0.0f);

        for (int k = K; k >= 0; --k) {
            float scale = lossScale(k);

            // prediction head at step k
            ws.scalarGrad.assign(1, scale * 2.0f * (ws.value[k] - sample.targetValues[k]) *
                                        (1.0f - ws.value[k] * ws.value[k]));
            ws.dPolicyLogits.resize(kActionSize);
            for (int a = 0; a < kActionSize; ++a) {
                // d(cross-entropy o softmax)/d(logit) = p - target
                ws.dPolicyLogits[a] = scale * (ws.policy[k][a] - sample.targetPolicies[k][a]);
            }

            fValue_.backwardInto(ws.predictionAct[k], ws.scalarGrad, ws.dPredictionHidden);
            fPolicy_.backwardInto(ws.predictionAct[k], ws.dPolicyLogits, ws.dFromPolicy);
            for (int i = 0; i < kHiddenSize; ++i) ws.dPredictionHidden[i] += ws.dFromPolicy[i];

            reluBackwardInto(ws.predictionPre[k], ws.dPredictionHidden, ws.dPredictionPre);
            fFc1_.backwardInto(ws.latent[k], ws.dPredictionPre, ws.dFromPrediction);
            for (int i = 0; i < kLatentSize; ++i) ws.dLatent[k][i] += ws.dFromPrediction[i];

            if (k > 0) {
                // dynamics step k-1 -> k
                minMaxNormalizeBackwardInto(ws.latentPre[k], ws.dLatent[k], ws.dLatentPre);
                gFc2_.backwardInto(ws.dynamicsAct[k - 1], ws.dLatentPre, ws.dDynamicsHidden);

                ws.scalarGrad.assign(1, scale * 2.0f * (ws.reward[k] - sample.targetRewards[k]) *
                                            (1.0f - ws.reward[k] * ws.reward[k]));
                gReward_.backwardInto(ws.dynamicsAct[k - 1], ws.scalarGrad, ws.dFromReward);
                for (int i = 0; i < kHiddenSize; ++i) ws.dDynamicsHidden[i] += ws.dFromReward[i];

                reluBackwardInto(ws.dynamicsPre[k - 1], ws.dDynamicsHidden, ws.dDynamicsPre);
                gFc1_.backwardInto(ws.dynamicsInput[k - 1], ws.dDynamicsPre, ws.dDynamicsInput);

                // The half gradient. Scaling what flows back into the
                // dynamics input keeps gradient magnitude from compounding
                // across the recurrence. See setDynamicsGradientScale.
                for (int i = 0; i < kLatentSize; ++i) {
                    ws.dLatent[k - 1][i] += dynamicsGradientScale_ * ws.dDynamicsInput[i];
                }
            } else {
                // representation network
                minMaxNormalizeBackwardInto(ws.latentPre[0], ws.dLatent[0], ws.dLatentPre);
                hFc2_.backwardInto(ws.hidden, ws.dLatentPre, ws.dHidden);
                reluBackwardInto(ws.hiddenPre, ws.dHidden, ws.dHiddenPre);
                hFc1_.backwardInto(ws.observation, ws.dHiddenPre, ws.dFromPrediction);
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
