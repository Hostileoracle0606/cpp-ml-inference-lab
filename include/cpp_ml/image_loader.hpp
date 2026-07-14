#pragma once

#include "cpp_ml/domain.hpp"

#include <filesystem>

namespace cpp_ml {

// A deliberately small, dependency-free image loader for portable builds.
// P6 (binary) and P3 (ASCII) PPM files are supported. Decoding is kept outside
// inference so a PNG/JPEG adapter can be added without changing the model core.
class FileImageLoader final {
public:
    [[nodiscard]] Image load(const std::filesystem::path& path) const;
};

}  // namespace cpp_ml
