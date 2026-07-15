#include "cpp_ml/domain.hpp"
#include "cpp_ml/inference_engine.hpp"
#include "cpp_ml/preprocessor.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using cpp_ml::Cifar10Preprocessor;
using cpp_ml::Image;
using cpp_ml::InferenceEngine;
using cpp_ml::ModelOutput;
using cpp_ml::Tensor;

constexpr std::size_t kChannels = 3;
constexpr std::size_t kHeight = 32;
constexpr std::size_t kWidth = 32;
constexpr std::size_t kValuesPerRow = kChannels * kHeight * kWidth;
constexpr std::size_t kClasses = 10;

struct Arguments {
    std::filesystem::path fixture_directory;
    std::filesystem::path trained_model;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

void expect_failure(const std::function<void()>& operation,
                    const std::vector<std::string_view>& concepts) {
    try {
        operation();
    } catch (const std::exception& error) {
        const std::string message = lowercase(error.what());
        const bool meaningful = std::any_of(
            concepts.begin(), concepts.end(), [&](std::string_view concept) {
                return message.find(concept) != std::string::npos;
            });
        require(meaningful, "failure diagnostic lacked an expected concept: " + message);
        return;
    }
    fail("operation unexpectedly succeeded");
}

Tensor prepared_rows(std::size_t row_count) {
    Tensor batch;
    batch.shape = {static_cast<std::int64_t>(row_count), 3, 32, 32};
    batch.values.reserve(row_count * kValuesPerRow);

    const Cifar10Preprocessor preprocessor;
    for (std::size_t row = 0; row < row_count; ++row) {
        Image image;
        image.width = kWidth;
        image.height = kHeight;
        image.channels = kChannels;
        image.pixels.resize(kValuesPerRow);
        for (std::size_t index = 0; index < image.pixels.size(); ++index) {
            image.pixels[index] = static_cast<std::uint8_t>(
                (index * 37U + row * 17U) % 256U);
        }
        const Tensor prepared = preprocessor.preprocess(image);
        require(prepared.shape == std::vector<std::int64_t>({1, 3, 32, 32}),
                "single-row preprocessing changed its public batch-one contract");
        batch.values.insert(batch.values.end(), prepared.values.begin(), prepared.values.end());
    }
    return batch;
}

Tensor select_row(const Tensor& batch, std::size_t row) {
    require(batch.shape.size() == 4, "cannot select a row from a non-NCHW tensor");
    require(row < static_cast<std::size_t>(batch.shape.front()),
            "row selection exceeded the batch");
    const auto begin = batch.values.begin() + static_cast<std::ptrdiff_t>(row * kValuesPerRow);
    Tensor selected;
    selected.shape = {1, 3, 32, 32};
    selected.values.assign(begin, begin + static_cast<std::ptrdiff_t>(kValuesPerRow));
    return selected;
}

InferenceEngine engine_for(const std::filesystem::path& model) {
    return InferenceEngine(cpp_ml::make_onnxruntime_backend(model));
}

void require_output_contract(const ModelOutput& output, std::size_t batch_size) {
    require(output.shape ==
                std::vector<std::int64_t>({static_cast<std::int64_t>(batch_size), 10}),
            "runtime output did not preserve [N,10] shape");
    require(output.logits.size() == batch_size * kClasses,
            "runtime output did not contain N*10 logits");
    require(std::all_of(output.logits.begin(), output.logits.end(),
                        [](float value) { return std::isfinite(value); }),
            "runtime output contained a non-finite logit");
}

std::size_t top_class(const float* logits) {
    return static_cast<std::size_t>(
        std::max_element(logits, logits + kClasses) - logits);
}

void test_dynamic_batch_sizes(const std::filesystem::path& fixtures) {
    auto engine = engine_for(fixtures / "valid.onnx");
    for (const std::size_t batch_size : {std::size_t{1}, std::size_t{8},
                                         std::size_t{256}}) {
        const ModelOutput output = engine.infer(prepared_rows(batch_size));
        require_output_contract(output, batch_size);
    }
}

void test_batch_bounds(const std::filesystem::path& fixtures) {
    auto engine = engine_for(fixtures / "valid.onnx");
    expect_failure(
        [&] { static_cast<void>(engine.infer(Tensor{{0, 3, 32, 32}, {}})); },
        {"batch", "positive"});
    expect_failure(
        [&] { static_cast<void>(engine.infer(Tensor{{257, 3, 32, 32}, {}})); },
        {"batch", "256"});
}

void test_fixed_batch_one_compatibility(const std::filesystem::path& fixtures) {
    auto engine = engine_for(fixtures / "fixed_batch_one.onnx");
    require_output_contract(engine.infer(prepared_rows(1)), 1);
    expect_failure([&] { static_cast<void>(engine.infer(prepared_rows(8))); },
                   {"batch", "fixed", "dimension"});
}

void test_invalid_batch_metadata(const std::filesystem::path& fixtures) {
    for (const auto& fixture : {
             "mixed_dynamic_input_fixed_output.onnx",
             "mixed_fixed_input_dynamic_output.onnx",
             "wrong_input_batch.onnx",
         }) {
        expect_failure(
            [&] {
                static_cast<void>(cpp_ml::make_onnxruntime_backend(fixtures / fixture));
            },
            {"batch", "dynamic", "fixed"});
    }
}

void test_runtime_output_shape_validation(const std::filesystem::path& fixtures) {
    for (const auto& fixture : {
             "runtime_batch_mismatch.onnx",
             "runtime_class_mismatch.onnx",
         }) {
        auto engine = engine_for(fixtures / fixture);
        expect_failure([&] { static_cast<void>(engine.infer(prepared_rows(8))); },
                       {"shape", "batch", "class", "count"});
    }
}

void test_row_ordering(const std::filesystem::path& fixtures) {
    const Tensor input = prepared_rows(8);
    auto engine = engine_for(fixtures / "row_identity_dynamic_batch.onnx");
    const ModelOutput output = engine.infer(input);
    require_output_contract(output, 8);

    for (std::size_t row = 0; row < 8; ++row) {
        for (std::size_t column = 0; column < kClasses; ++column) {
            const float expected = input.values[row * kValuesPerRow + column];
            const float actual = output.logits[row * kClasses + column];
            require(std::abs(actual - expected) <= 1.0e-6F,
                    "runtime output did not preserve deterministic row-major mapping");
        }
    }
}

void test_trained_batch_matches_single_rows(const std::filesystem::path& model) {
    const Tensor input = prepared_rows(8);
    auto engine = engine_for(model);
    const ModelOutput batched = engine.infer(input);
    require_output_contract(batched, 8);

    for (std::size_t row = 0; row < 8; ++row) {
        const ModelOutput single = engine.infer(select_row(input, row));
        require_output_contract(single, 1);
        float maximum_difference = 0.0F;
        for (std::size_t column = 0; column < kClasses; ++column) {
            maximum_difference = std::max(
                maximum_difference,
                std::abs(batched.logits[row * kClasses + column] -
                         single.logits[column]));
        }
        require(maximum_difference <= 1.0e-5F,
                "trained-model batch/single maximum logit difference exceeded 1e-5");
        require(top_class(batched.logits.data() + row * kClasses) ==
                    top_class(single.logits.data()),
                "trained-model batch/single top classes differed");
    }
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if ((option == "--fixtures" || option == "--trained-model") && index + 1 < argc) {
            const std::filesystem::path value = argv[++index];
            if (option == "--fixtures") {
                arguments.fixture_directory = value;
            } else {
                arguments.trained_model = value;
            }
        } else {
            fail("usage: cpp_ml_batch_ort_tests [--fixtures DIR] [--trained-model MODEL]");
        }
    }
    require(!arguments.fixture_directory.empty() || !arguments.trained_model.empty(),
            "at least one batch test input is required");
    return arguments;
}

template <typename Function>
void run_test(std::string_view name, Function&& function, std::size_t& failures) {
    try {
        std::forward<Function>(function)();
        std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        std::size_t failures = 0;
        if (!arguments.fixture_directory.empty()) {
            run_test("dynamic batch sizes 1, 8, and 256",
                     [&] { test_dynamic_batch_sizes(arguments.fixture_directory); },
                     failures);
            run_test("batch bounds 0 and 257",
                     [&] { test_batch_bounds(arguments.fixture_directory); }, failures);
            run_test("fixed batch-one compatibility",
                     [&] { test_fixed_batch_one_compatibility(arguments.fixture_directory); },
                     failures);
            run_test("invalid fixed and mixed batch metadata",
                     [&] { test_invalid_batch_metadata(arguments.fixture_directory); },
                     failures);
            run_test("runtime output shape validation",
                     [&] { test_runtime_output_shape_validation(arguments.fixture_directory); },
                     failures);
            run_test("deterministic row ordering",
                     [&] { test_row_ordering(arguments.fixture_directory); }, failures);
        }
        if (!arguments.trained_model.empty()) {
            run_test("trained-model batch/single numerical parity",
                     [&] { test_trained_batch_matches_single_rows(arguments.trained_model); },
                     failures);
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] batch test setup: " << error.what() << '\n';
        return 1;
    }
}
