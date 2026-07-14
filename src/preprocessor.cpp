#include "cpp_ml/preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cpp_ml {
namespace {

float source_coordinate(std::size_t output_index, std::size_t output_size,
                        std::size_t input_size) {
    return (static_cast<float>(output_index) + 0.5F) *
               static_cast<float>(input_size) / static_cast<float>(output_size) -
           0.5F;
}

float sample_bilinear(const Image& image, std::size_t out_x, std::size_t out_y,
                      std::size_t channel) {
    const float raw_x = source_coordinate(out_x, Cifar10Preprocessor::kInputWidth,
                                          image.width);
    const float raw_y = source_coordinate(out_y, Cifar10Preprocessor::kInputHeight,
                                          image.height);
    const float x = std::clamp(raw_x, 0.0F, static_cast<float>(image.width - 1));
    const float y = std::clamp(raw_y, 0.0F, static_cast<float>(image.height - 1));
    const auto x0 = static_cast<std::size_t>(std::floor(x));
    const auto y0 = static_cast<std::size_t>(std::floor(y));
    const auto x1 = std::min(x0 + 1, image.width - 1);
    const auto y1 = std::min(y0 + 1, image.height - 1);
    const float x_weight = x - static_cast<float>(x0);
    const float y_weight = y - static_cast<float>(y0);

    const auto at = [&](std::size_t px, std::size_t py) {
        return static_cast<float>(
            image.pixels[(py * image.width + px) * image.channels + channel]);
    };
    const float top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * x_weight;
    const float bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * x_weight;
    return top + (bottom - top) * y_weight;
}

}  // namespace

Tensor Cifar10Preprocessor::preprocess(const Image& image) const {
    image.validate();
    if (image.channels != kInputChannels) {
        throw std::invalid_argument("CIFAR-10 preprocessing requires an RGB image");
    }

    Tensor result;
    result.shape = {1, static_cast<std::int64_t>(kInputChannels),
                    static_cast<std::int64_t>(kInputHeight),
                    static_cast<std::int64_t>(kInputWidth)};
    result.values.resize(kInputChannels * kInputHeight * kInputWidth);

    const bool same_size = image.width == kInputWidth && image.height == kInputHeight;
    for (std::size_t channel = 0; channel < kInputChannels; ++channel) {
        for (std::size_t y = 0; y < kInputHeight; ++y) {
            for (std::size_t x = 0; x < kInputWidth; ++x) {
                const float sample = same_size
                                         ? static_cast<float>(image.pixels[
                                               (y * image.width + x) * image.channels +
                                               channel])
                                         : sample_bilinear(image, x, y, channel);
                const float scaled = sample / 255.0F;
                const auto destination =
                    channel * kInputHeight * kInputWidth + y * kInputWidth + x;
                result.values[destination] = (scaled - kMean[channel]) / kStd[channel];
            }
        }
    }
    result.validate();
    return result;
}

}  // namespace cpp_ml
