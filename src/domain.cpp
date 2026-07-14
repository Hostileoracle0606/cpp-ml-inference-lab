#include "cpp_ml/domain.hpp"

#include <limits>
#include <stdexcept>

namespace cpp_ml {
namespace {

std::size_t checked_product(std::size_t left, std::size_t right,
                            const char* description) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::invalid_argument(std::string(description) + " size overflows");
    }
    return left * right;
}

}  // namespace

void Image::validate() const {
    if (width == 0 || height == 0 || channels == 0) {
        throw std::invalid_argument("image dimensions and channel count must be positive");
    }
    const auto area = checked_product(width, height, "image");
    const auto expected = checked_product(area, channels, "image");
    if (pixels.size() != expected) {
        throw std::invalid_argument("image pixel buffer does not match its dimensions");
    }
}

std::size_t Tensor::element_count() const {
    std::size_t count = 1;
    for (const auto dimension : shape) {
        if (dimension <= 0) {
            throw std::invalid_argument("tensor dimensions must be positive");
        }
        count = checked_product(count, static_cast<std::size_t>(dimension), "tensor");
    }
    return count;
}

void Tensor::validate() const {
    if (shape.empty()) {
        throw std::invalid_argument("tensor shape must not be empty");
    }
    if (values.size() != element_count()) {
        throw std::invalid_argument("tensor value buffer does not match its shape");
    }
}

}  // namespace cpp_ml
