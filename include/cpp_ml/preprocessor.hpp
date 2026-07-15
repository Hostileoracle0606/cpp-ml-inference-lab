#pragma once

#include "cpp_ml/domain.hpp"

#include <array>
#include <cstddef>

namespace cpp_ml {

class Cifar10Preprocessor final {
public:
    static constexpr std::size_t kInputWidth = 32;
    static constexpr std::size_t kInputHeight = 32;
    static constexpr std::size_t kInputChannels = 3;
    static constexpr std::array<float, kInputChannels> kMean = {
        0.4914F, 0.4822F, 0.4465F};
    static constexpr std::array<float, kInputChannels> kStd = {
        0.2470F, 0.2435F, 0.2616F};

    // Converts interleaved RGB uint8 pixels to normalized NCHW float values.
    // Images not already 32x32 are resized using bilinear interpolation.
    [[nodiscard]] Tensor preprocess(const Image& image) const;
};

}  // namespace cpp_ml
