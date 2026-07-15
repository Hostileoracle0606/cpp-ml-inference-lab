#include "cpp_ml/inference_engine.hpp"
#include "cpp_ml/pipeline.hpp"
#include "cpp_ml/preprocessor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
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
    bool paired_batch = false;
    bool warmup_was_set = false;
    bool iterations_was_set = false;
    std::size_t paired_run = 0;
    bool paired_run_was_set = false;
    std::string json_output;
};

constexpr std::size_t kPairedBatchSize = 8;
constexpr std::size_t kPairedWarmup = 20;
constexpr std::size_t kPairedIterations = 200;

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
        if (option == "--paired-batch") {
            result.paired_batch = true;
            continue;
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + std::string(option));
        }
        if (option == "--model") {
            result.model = argv[++index];
        } else if (option == "--warmup") {
            result.warmup = parse_positive(argv[++index], option);
            result.warmup_was_set = true;
        } else if (option == "--iterations") {
            result.iterations = parse_positive(argv[++index], option);
            result.iterations_was_set = true;
        } else if (option == "--paired-run") {
            result.paired_run = parse_positive(argv[++index], option);
            result.paired_run_was_set = true;
        } else if (option == "--json-out") {
            result.json_output = argv[++index];
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(option));
        }
    }

    if (result.paired_batch) {
        if (!result.paired_run_was_set || result.paired_run > 10) {
            throw std::invalid_argument(
                "paired batch mode requires --paired-run with a value from 1 to 10");
        }
        if (result.json_output.empty()) {
            throw std::invalid_argument("paired batch mode requires --json-out <path>");
        }
        if ((result.warmup_was_set && result.warmup != kPairedWarmup) ||
            (result.iterations_was_set && result.iterations != kPairedIterations)) {
            throw std::invalid_argument(
                "paired batch mode is frozen at --warmup 20 --iterations 200");
        }
        result.warmup = kPairedWarmup;
        result.iterations = kPairedIterations;
    } else if (result.paired_run_was_set || !result.json_output.empty()) {
        throw std::invalid_argument(
            "--paired-run and --json-out require --paired-batch");
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

double paired_items_per_second(const std::vector<double>& samples) {
    if (samples.size() != kPairedIterations) {
        throw std::invalid_argument("paired sample count does not match the frozen experiment");
    }
    const auto total_ms = std::accumulate(samples.begin(), samples.end(), 0.0);
    if (!std::isfinite(total_ms) || total_ms <= 0.0) {
        throw std::runtime_error("paired benchmark produced an invalid total duration");
    }
    const auto item_count = static_cast<double>(kPairedBatchSize * samples.size());
    return 1000.0 * item_count / total_ms;
}

void report_paired_mode(std::string_view name, const std::vector<double>& samples) {
    const auto total_ms = std::accumulate(samples.begin(), samples.end(), 0.0);
    const auto mean_group_ms = total_ms / static_cast<double>(samples.size());
    std::cout << name << "\n"
              << "  mean_group:       " << mean_group_ms << " ms\n"
              << "  p50_group:        " << percentile(samples, 0.50) << " ms\n"
              << "  p95_group:        " << percentile(samples, 0.95) << " ms\n"
              << "  amortized:        " << mean_group_ms / kPairedBatchSize
              << " ms/item\n"
              << "  item_throughput:  " << paired_items_per_second(samples)
              << " items/s\n";
}

void write_json_string(std::ostream& output, std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (character < 0x20U) {
                    output << "\\u00" << hex[character >> 4U]
                           << hex[character & 0x0fU];
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    output << '"';
}

void write_json_samples(std::ostream& output, const std::vector<double>& samples) {
    output << '[';
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << samples[index];
    }
    output << ']';
}

void write_paired_json(const Arguments& arguments,
                       const std::vector<double>& serial_samples,
                       const std::vector<double>& batch_samples) {
    std::ofstream output(arguments.json_output, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open paired benchmark JSON output: " +
                                 arguments.json_output);
    }

    const auto serial_throughput = paired_items_per_second(serial_samples);
    const auto batch_throughput = paired_items_per_second(batch_samples);
    const auto order = arguments.paired_run % 2U == 1U
                           ? "serial_then_batch"
                           : "batch_then_serial";
    output << std::setprecision(17)
           << "{\n  \"schema\": \"cpp_ml.paired_batch.v1\",\n"
           << "  \"run\": " << arguments.paired_run << ",\n"
           << "  \"order\": \"" << order << "\",\n"
           << "  \"batch_size\": " << kPairedBatchSize << ",\n"
           << "  \"warmup_workloads_per_mode\": " << kPairedWarmup << ",\n"
           << "  \"measured_workloads_per_mode\": " << kPairedIterations << ",\n"
           << "  \"input_recipe\": \"byte_i_row_r=(i*37+r*17)%256\",\n"
           << "  \"model\": ";
    write_json_string(output, arguments.model);
    output << ",\n  \"build\": {\n"
           << "    \"type\": ";
    write_json_string(output, CPP_ML_BUILD_TYPE);
    output << ",\n    \"system\": ";
    write_json_string(output,
                      std::string(CPP_ML_SYSTEM_NAME) + "/" + CPP_ML_SYSTEM_PROCESSOR);
    output << ",\n    \"compiler\": ";
    write_json_string(output,
                      std::string(CPP_ML_COMPILER_ID) + " " + CPP_ML_COMPILER_VERSION);
    output << ",\n    \"cxx\": " << __cplusplus << "\n  },\n"
           << "  \"serial_eight\": {\n"
           << "    \"runtime_calls_per_group\": 8,\n"
           << "    \"items_per_second\": " << serial_throughput << ",\n"
           << "    \"group_latency_ms\": ";
    write_json_samples(output, serial_samples);
    output << "\n  },\n  \"batch_eight\": {\n"
           << "    \"runtime_calls_per_group\": 1,\n"
           << "    \"items_per_second\": " << batch_throughput << ",\n"
           << "    \"group_latency_ms\": ";
    write_json_samples(output, batch_samples);
    output << "\n  },\n"
           << "  \"batch_to_serial_items_per_second_ratio\": "
           << batch_throughput / serial_throughput << "\n}\n";
    if (!output) {
        throw std::runtime_error("failed to write paired benchmark JSON output: " +
                                 arguments.json_output);
    }
}

cpp_ml::Image make_paired_image(std::size_t row) {
    cpp_ml::Image image;
    image.width = 32;
    image.height = 32;
    image.channels = 3;
    image.pixels.resize(image.width * image.height * image.channels);
    for (std::size_t index = 0; index < image.pixels.size(); ++index) {
        image.pixels[index] =
            static_cast<std::uint8_t>((index * 37U + row * 17U) % 256U);
    }
    return image;
}

int run_paired_benchmark(const Arguments& arguments) {
    if (arguments.model.empty()) {
        throw std::invalid_argument("paired batch mode requires --model <path.onnx>");
    }

    cpp_ml::Cifar10Preprocessor preprocessor;
    std::vector<cpp_ml::Tensor> rows;
    rows.reserve(kPairedBatchSize);
    cpp_ml::Tensor batch;
    batch.shape = {static_cast<std::int64_t>(kPairedBatchSize), 3, 32, 32};
    batch.values.reserve(kPairedBatchSize * 3U * 32U * 32U);
    for (std::size_t row = 0; row < kPairedBatchSize; ++row) {
        rows.push_back(preprocessor.preprocess(make_paired_image(row)));
        batch.values.insert(batch.values.end(), rows.back().values.begin(),
                            rows.back().values.end());
    }
    batch.validate();

    cpp_ml::InferenceEngine engine(cpp_ml::make_onnxruntime_backend(arguments.model));
    cpp_ml::ModelOutput serial_output;
    cpp_ml::ModelOutput batch_output;
    const auto serial_operation = [&] {
        for (const auto& row : rows) {
            serial_output = engine.infer(row);
        }
    };
    const auto batch_operation = [&] { batch_output = engine.infer(batch); };

    std::vector<double> serial_samples;
    std::vector<double> batch_samples;
    if (arguments.paired_run % 2U == 1U) {
        serial_samples = measure(kPairedWarmup, kPairedIterations, serial_operation);
        batch_samples = measure(kPairedWarmup, kPairedIterations, batch_operation);
    } else {
        batch_samples = measure(kPairedWarmup, kPairedIterations, batch_operation);
        serial_samples = measure(kPairedWarmup, kPairedIterations, serial_operation);
    }

    write_paired_json(arguments, serial_samples, batch_samples);
    const auto order = arguments.paired_run % 2U == 1U
                           ? "serial_then_batch"
                           : "batch_then_serial";
    std::cout << std::fixed << std::setprecision(4)
              << "paired_runtime_batch8\n"
              << "  run:              " << arguments.paired_run << "\n"
              << "  order:            " << order << "\n"
              << "  warmup:           " << kPairedWarmup << " workloads/mode\n"
              << "  iterations:       " << kPairedIterations << " workloads/mode\n";
    report_paired_mode("serial_eight", serial_samples);
    report_paired_mode("batch_eight", batch_samples);
    std::cout << "paired_ratio: "
              << paired_items_per_second(batch_samples) /
                     paired_items_per_second(serial_samples)
              << "\n"
              << "machine_output: " << arguments.json_output << "\n";
    return serial_output.logits.empty() || batch_output.logits.empty() ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        if (arguments.paired_batch) {
            return run_paired_benchmark(arguments);
        }
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
