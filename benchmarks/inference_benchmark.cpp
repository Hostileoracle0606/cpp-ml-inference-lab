#include "cpp_ml/inference_engine.hpp"
#include "cpp_ml/pipeline.hpp"
#include "cpp_ml/preprocessor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef CPP_ML_BUILD_TYPE
#define CPP_ML_BUILD_TYPE "unknown"
#endif
#ifndef CPP_ML_SYSTEM_NAME
#define CPP_ML_SYSTEM_NAME "unknown"
#endif
#ifndef CPP_ML_SYSTEM_PROCESSOR
#define CPP_ML_SYSTEM_PROCESSOR "unknown"
#endif
#ifndef CPP_ML_COMPILER_ID
#define CPP_ML_COMPILER_ID "unknown"
#endif
#ifndef CPP_ML_COMPILER_VERSION
#define CPP_ML_COMPILER_VERSION "unknown"
#endif

namespace {

struct Arguments {
    std::string model;
    std::size_t warmup = 10;
    std::size_t iterations = 100;
};

std::size_t parse_positive(const char* text, std::string_view option) {
    std::size_t consumed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid value for " + std::string(option));
    }
    if (text[consumed] != '\0' || value == 0 ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("invalid value for " + std::string(option));
    }
    return static_cast<std::size_t>(value);
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + std::string(option));
        }
        if (option == "--model") {
            result.model = argv[++index];
        } else if (option == "--warmup") {
            result.warmup = parse_positive(argv[++index], option);
        } else if (option == "--iterations") {
            result.iterations = parse_positive(argv[++index], option);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(option));
        }
    }
    return result;
}

double percentile(std::vector<double> samples, double quantile) {
    if (samples.empty()) {
        throw std::invalid_argument("cannot compute percentile of no samples");
    }
    std::sort(samples.begin(), samples.end());
    const auto rank = static_cast<std::size_t>(
        std::ceil(quantile * static_cast<double>(samples.size())));
    return samples[std::max<std::size_t>(1, rank) - 1];
}

void report(std::string_view name, const std::vector<double>& samples) {
    const double total =
        std::accumulate(samples.begin(), samples.end(), 0.0);
    const double throughput = total == 0.0
                                  ? 0.0
                                  : 1000.0 * static_cast<double>(samples.size()) / total;
    std::cout << name << "\n"
              << "  mean:       " << total / static_cast<double>(samples.size())
              << " ms\n"
              << "  p50:        " << percentile(samples, 0.50) << " ms\n"
              << "  p95:        " << percentile(samples, 0.95) << " ms\n"
              << "  throughput: " << throughput << " operations/s\n";
}

template <typename Operation>
std::vector<double> measure(std::size_t warmup, std::size_t iterations,
                            Operation operation) {
    for (std::size_t index = 0; index < warmup; ++index) {
        operation();
    }
    std::vector<double> samples;
    samples.reserve(iterations);
    for (std::size_t index = 0; index < iterations; ++index) {
        const auto started = std::chrono::steady_clock::now();
        operation();
        const auto finished = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(finished - started).count());
    }
    return samples;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        cpp_ml::Image image;
        image.width = 32;
        image.height = 32;
        image.channels = 3;
        image.pixels.resize(image.width * image.height * image.channels);
        for (std::size_t index = 0; index < image.pixels.size(); ++index) {
            image.pixels[index] = static_cast<std::uint8_t>((index * 37U) % 256U);
        }

        cpp_ml::Cifar10Preprocessor preprocessor;
        cpp_ml::Tensor tensor;
        const auto preprocessing = measure(
            arguments.warmup, arguments.iterations,
            [&] { tensor = preprocessor.preprocess(image); });

        std::cout << std::fixed << std::setprecision(4)
                  << "build_type: " << CPP_ML_BUILD_TYPE << "\n"
                  << "system: " << CPP_ML_SYSTEM_NAME << '/'
                  << CPP_ML_SYSTEM_PROCESSOR << "\n"
                  << "compiler: " << CPP_ML_COMPILER_ID << ' '
                  << CPP_ML_COMPILER_VERSION << "\n"
                  << "cxx: " << __cplusplus << "\n"
                  << "warmup: " << arguments.warmup << "\n"
                  << "iterations: " << arguments.iterations << "\n";
        report("preprocessing", preprocessing);

        if (arguments.model.empty()) {
            std::cout << "inference: skipped (pass --model <path.onnx>)\n";
            return 0;
        }

        cpp_ml::InferenceEngine engine(
            cpp_ml::make_onnxruntime_backend(arguments.model));
        cpp_ml::ModelOutput output;
        const auto inference = measure(arguments.warmup, arguments.iterations,
                                       [&] { output = engine.infer(tensor); });
        report("runtime_only", inference);

        cpp_ml::InferencePipeline pipeline(std::move(engine));
        cpp_ml::Prediction prediction;
        const auto end_to_end = measure(
            arguments.warmup, arguments.iterations,
            [&] { prediction = pipeline.predict(image); });
        report("end_to_end", end_to_end);
        return output.logits.empty() || prediction.probabilities.empty() ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
