#include "cpp_ml/image_loader.hpp"

#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace cpp_ml {
namespace {

std::string read_token(std::istream& input) {
    std::string token;
    char character = '\0';

    while (input.get(character)) {
        if (std::isspace(static_cast<unsigned char>(character)) != 0) {
            continue;
        }
        if (character == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        token.push_back(character);
        break;
    }

    while (input.get(character)) {
        if (std::isspace(static_cast<unsigned char>(character)) != 0) {
            break;
        }
        if (character == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            break;
        }
        token.push_back(character);
    }

    if (token.empty()) {
        throw std::runtime_error("unexpected end of PPM header");
    }
    return token;
}

std::size_t parse_unsigned(const std::string& token, const char* field,
                           bool allow_zero) {
    std::size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(token, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid PPM ") + field);
    }
    if (consumed != token.size() || (!allow_zero && parsed == 0) ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string("invalid PPM ") + field);
    }
    return static_cast<std::size_t>(parsed);
}

std::size_t parse_positive(const std::string& token, const char* field) {
    return parse_unsigned(token, field, false);
}

std::uint8_t scale_sample(std::size_t sample, std::size_t maximum) {
    if (sample > maximum) {
        throw std::runtime_error("PPM sample exceeds declared maximum");
    }
    return static_cast<std::uint8_t>((sample * 255U + maximum / 2U) / maximum);
}

}  // namespace

Image FileImageLoader::load(const std::filesystem::path& path) const {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open image: " + path.string());
    }

    const auto magic = read_token(input);
    if (magic != "P6" && magic != "P3") {
        throw std::runtime_error(
            "unsupported image format for '" + path.string() +
            "' (portable build supports P6/P3 PPM)");
    }

    Image image;
    image.width = parse_positive(read_token(input), "width");
    image.height = parse_positive(read_token(input), "height");
    image.channels = 3;
    const auto maximum = parse_positive(read_token(input), "maximum sample value");
    if (maximum > 255) {
        throw std::runtime_error("16-bit PPM images are not supported");
    }
    constexpr std::size_t kMaximumDimension = 16'384;
    constexpr std::size_t kMaximumPixels = 16'777'216;
    if (image.width > kMaximumDimension || image.height > kMaximumDimension) {
        throw std::runtime_error("PPM image dimension exceeds safety limit");
    }
    if (image.width > std::numeric_limits<std::size_t>::max() / image.height ||
        image.width * image.height > std::numeric_limits<std::size_t>::max() / 3U) {
        throw std::runtime_error("PPM image dimensions overflow");
    }
    if (image.width * image.height > kMaximumPixels) {
        throw std::runtime_error("PPM decoded pixel count exceeds safety limit");
    }
    const auto sample_count = image.width * image.height * 3U;
    image.pixels.resize(sample_count);

    if (magic == "P6") {
        input.read(reinterpret_cast<char*>(image.pixels.data()),
                   static_cast<std::streamsize>(sample_count));
        if (input.gcount() != static_cast<std::streamsize>(sample_count)) {
            throw std::runtime_error("truncated P6 PPM pixel data");
        }
        if (maximum != 255) {
            for (auto& sample : image.pixels) {
                sample = scale_sample(sample, maximum);
            }
        }
    } else {
        for (auto& sample : image.pixels) {
            sample = scale_sample(
                parse_unsigned(read_token(input), "pixel sample", true), maximum);
        }
    }

    image.validate();
    return image;
}

}  // namespace cpp_ml
