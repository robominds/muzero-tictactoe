#include "mz/network.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

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

} // namespace mz

namespace mz {
// Replaced in full by Task 6.
MuZeroNetwork::Losses MuZeroNetwork::trainStep(const std::vector<UnrolledSample>&, float) {
    return Losses{};
}
} // namespace mz
