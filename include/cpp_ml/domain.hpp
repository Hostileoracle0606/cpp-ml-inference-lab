#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cpp_ml {

struct Image {
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t channels = 0;
    std::vector<std::uint8_t> pixels;

    void validate() const;
};

struct Tensor {
    std::vector<std::int64_t> shape;
    std::vector<float> values;

    [[nodiscard]] std::size_t element_count() const;
    void validate() const;
};

struct ModelOutput {
    std::vector<float> logits;
    double inference_ms = 0.0;
};

struct Prediction {
    std::size_t class_index = 0;
    std::string label;
    float confidence = 0.0F;
    std::vector<float> probabilities;
    double preprocessing_ms = 0.0;
    double inference_ms = 0.0;
    double latency_ms = 0.0;
};

}  // namespace cpp_ml
