#pragma once

#include "cpp_ml/domain.hpp"

#include <string>
#include <vector>

namespace cpp_ml {

class SoftmaxDecoder final {
public:
    SoftmaxDecoder();
    explicit SoftmaxDecoder(std::vector<std::string> labels);

    [[nodiscard]] Prediction decode(const std::vector<float>& logits) const;
    [[nodiscard]] const std::vector<std::string>& labels() const noexcept;

private:
    std::vector<std::string> labels_;
};

[[nodiscard]] std::vector<std::string> cifar10_labels();

}  // namespace cpp_ml
