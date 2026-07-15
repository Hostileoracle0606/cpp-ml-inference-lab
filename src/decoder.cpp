#include "cpp_ml/decoder.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cpp_ml {

std::vector<std::string> cifar10_labels() {
    return {"airplane", "automobile", "bird", "cat", "deer",
            "dog",      "frog",       "horse", "ship", "truck"};
}

SoftmaxDecoder::SoftmaxDecoder() : SoftmaxDecoder(cifar10_labels()) {}

SoftmaxDecoder::SoftmaxDecoder(std::vector<std::string> labels)
    : labels_(std::move(labels)) {
    if (labels_.empty() ||
        std::any_of(labels_.begin(), labels_.end(),
                    [](const std::string& label) { return label.empty(); })) {
        throw std::invalid_argument("decoder labels must be non-empty");
    }
}

Prediction SoftmaxDecoder::decode(const std::vector<float>& logits) const {
    if (logits.size() != labels_.size()) {
        throw std::invalid_argument("logit count does not match decoder label count");
    }
    if (!std::all_of(logits.begin(), logits.end(),
                     [](float value) { return std::isfinite(value); })) {
        throw std::invalid_argument("logits must all be finite");
    }

    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probabilities(logits.size());
    std::transform(logits.begin(), logits.end(), probabilities.begin(),
                   [maximum](float value) { return std::exp(value - maximum); });
    const float denominator =
        std::accumulate(probabilities.begin(), probabilities.end(), 0.0F);
    for (auto& probability : probabilities) {
        probability /= denominator;
    }

    const auto best = std::max_element(probabilities.begin(), probabilities.end());
    const auto index = static_cast<std::size_t>(
        std::distance(probabilities.begin(), best));
    Prediction prediction;
    prediction.class_index = index;
    prediction.label = labels_[index];
    prediction.confidence = *best;
    prediction.probabilities = std::move(probabilities);
    return prediction;
}

const std::vector<std::string>& SoftmaxDecoder::labels() const noexcept {
    return labels_;
}

}  // namespace cpp_ml
