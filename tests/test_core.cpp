#include "test_harness.hpp"

#include "cpp_ml/decoder.hpp"
#include "cpp_ml/domain.hpp"
#include "cpp_ml/image_loader.hpp"
#include "cpp_ml/inference_engine.hpp"
#include "cpp_ml/pipeline.hpp"
#include "cpp_ml/preprocessor.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using cpp_ml::Cifar10Preprocessor;
using cpp_ml::Image;
using cpp_ml::InferenceEngine;
using cpp_ml::InferencePipeline;
using cpp_ml::IInferenceBackend;
using cpp_ml::ModelOutput;
using cpp_ml::SoftmaxDecoder;
using cpp_ml::Tensor;

constexpr float kTolerance = 1.0e-5F;

Image constant_image(
    std::size_t width,
    std::size_t height,
    const std::array<std::uint8_t, 3>& rgb) {
    Image image;
    image.width = width;
    image.height = height;
    image.channels = 3;
    image.pixels.reserve(width * height * 3);
    for (std::size_t i = 0; i < width * height; ++i) {
        image.pixels.insert(image.pixels.end(), rgb.begin(), rgb.end());
    }
    return image;
}

float normalized(std::uint8_t value, std::size_t channel) {
    return (static_cast<float>(value) / 255.0F - Cifar10Preprocessor::kMean[channel]) /
           Cifar10Preprocessor::kStd[channel];
}

ModelOutput model_output(std::vector<float> logits,
                         std::vector<std::int64_t> shape,
                         double inference_ms = 0.0) {
    ModelOutput output;
    output.logits = std::move(logits);
    output.inference_ms = inference_ms;
    output.shape = std::move(shape);
    return output;
}

Tensor runtime_tensor(std::int64_t batch_size) {
    Tensor tensor;
    tensor.shape = {batch_size, 3, 32, 32};
    if (batch_size > 0 && batch_size <= 256) {
        tensor.values.resize(static_cast<std::size_t>(batch_size) * 3U * 32U * 32U);
    }
    return tensor;
}

class TemporaryFile final {
public:
    explicit TemporaryFile(std::string suffix)
        : path_(std::filesystem::temp_directory_path() /
                ("cpp_ml_test_" + std::to_string(++counter_) + std::move(suffix))) {}

    ~TemporaryFile() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    inline static std::size_t counter_ = 0;
    std::filesystem::path path_;
};

struct BackendState {
    std::size_t calls = 0;
    Tensor last_input;
};

class RecordingBackend final : public IInferenceBackend {
public:
    RecordingBackend(std::shared_ptr<BackendState> state, ModelOutput output)
        : state_(std::move(state)), output_(std::move(output)) {}

    ModelOutput run(const Tensor& input) override {
        ++state_->calls;
        state_->last_input = input;
        return output_;
    }

private:
    std::shared_ptr<BackendState> state_;
    ModelOutput output_;
};

TEST_CASE(image_and_tensor_values_validate_their_invariants) {
    Image valid = constant_image(2, 2, {0, 127, 255});
    valid.validate();

    Image wrong_size = valid;
    wrong_size.pixels.pop_back();
    EXPECT_THROW(wrong_size.validate(), std::invalid_argument);

    Image zero_width = valid;
    zero_width.width = 0;
    EXPECT_THROW(zero_width.validate(), std::invalid_argument);

    Tensor tensor{{1, 3, 2, 2}, std::vector<float>(12, 0.0F)};
    EXPECT_EQ(tensor.element_count(), std::size_t{12});
    tensor.validate();

    Tensor wrong_elements = tensor;
    wrong_elements.values.pop_back();
    EXPECT_THROW(wrong_elements.validate(), std::invalid_argument);

    Tensor invalid_dimension{{1, 3, -1, 2}, {}};
    EXPECT_THROW(invalid_dimension.validate(), std::invalid_argument);

    Tensor zero_batch{{0, 3, 32, 32}, {}};
    EXPECT_THROW(zero_batch.validate(), std::invalid_argument);

    Tensor overflowing{{std::numeric_limits<std::int64_t>::max(), 3, 32, 32}, {}};
    EXPECT_THROW(overflowing.element_count(), std::invalid_argument);
}

TEST_CASE(model_output_validates_shape_count_finiteness_and_overflow) {
    ModelOutput batch_output = model_output(std::vector<float>(80, 0.5F), {8, 10}, 1.25);
    EXPECT_EQ(batch_output.element_count(), std::size_t{80});
    batch_output.validate();

    ModelOutput empty_shape = model_output({1.0F}, {});
    EXPECT_THROW(empty_shape.validate(), std::invalid_argument);

    ModelOutput zero_dimension = model_output({}, {0, 10});
    EXPECT_THROW(zero_dimension.validate(), std::invalid_argument);

    ModelOutput wrong_count = model_output(std::vector<float>(79, 0.0F), {8, 10});
    EXPECT_THROW(wrong_count.validate(), std::invalid_argument);

    ModelOutput non_finite = model_output(
        {std::numeric_limits<float>::quiet_NaN()}, {1, 1});
    EXPECT_THROW(non_finite.validate(), std::invalid_argument);

    ModelOutput overflowing = model_output(
        {}, {std::numeric_limits<std::int64_t>::max(), 3});
    EXPECT_THROW(overflowing.element_count(), std::invalid_argument);
}

TEST_CASE(preprocessor_preserves_contract_shape_and_hwc_to_chw_channel_order) {
    Image image = constant_image(32, 32, {0, 127, 255});
    const Tensor tensor = Cifar10Preprocessor{}.preprocess(image);

    EXPECT_EQ(tensor.shape, (std::vector<std::int64_t>{1, 3, 32, 32}));
    EXPECT_EQ(tensor.values.size(), std::size_t{3072});
    EXPECT_NEAR(tensor.values[0], normalized(0, 0), kTolerance);
    EXPECT_NEAR(tensor.values[1023], normalized(0, 0), kTolerance);
    EXPECT_NEAR(tensor.values[1024], normalized(127, 1), kTolerance);
    EXPECT_NEAR(tensor.values[2048], normalized(255, 2), kTolerance);
}

TEST_CASE(preprocessor_normalizes_pixel_boundaries_with_documented_constants) {
    EXPECT_NEAR(Cifar10Preprocessor::kMean[0], 0.4914F, 1.0e-7F);
    EXPECT_NEAR(Cifar10Preprocessor::kMean[1], 0.4822F, 1.0e-7F);
    EXPECT_NEAR(Cifar10Preprocessor::kMean[2], 0.4465F, 1.0e-7F);
    EXPECT_NEAR(Cifar10Preprocessor::kStd[0], 0.2470F, 1.0e-7F);
    EXPECT_NEAR(Cifar10Preprocessor::kStd[1], 0.2435F, 1.0e-7F);
    EXPECT_NEAR(Cifar10Preprocessor::kStd[2], 0.2616F, 1.0e-7F);

    Image image = constant_image(1, 1, {0, 255, 0});
    const Tensor tensor = Cifar10Preprocessor{}.preprocess(image);
    EXPECT_EQ(tensor.values.size(), std::size_t{3072});
    EXPECT_NEAR(tensor.values.front(), normalized(0, 0), kTolerance);
    EXPECT_NEAR(tensor.values[1024], normalized(255, 1), kTolerance);
    EXPECT_NEAR(tensor.values[2048], normalized(0, 2), kTolerance);
    EXPECT_NEAR(tensor.values.back(), normalized(0, 2), kTolerance);
}

TEST_CASE(preprocessor_rejects_invalid_images) {
    Image wrong_channels = constant_image(32, 32, {1, 2, 3});
    wrong_channels.channels = 1;
    EXPECT_THROW(Cifar10Preprocessor{}.preprocess(wrong_channels), std::invalid_argument);

    Image truncated = constant_image(32, 32, {1, 2, 3});
    truncated.pixels.resize(10);
    EXPECT_THROW(Cifar10Preprocessor{}.preprocess(truncated), std::invalid_argument);
}

TEST_CASE(softmax_is_stable_and_decodes_argmax_with_cifar10_labels) {
    const auto labels = cpp_ml::cifar10_labels();
    EXPECT_EQ(labels.size(), std::size_t{10});
    EXPECT_EQ(labels[0], std::string{"airplane"});
    EXPECT_EQ(labels[3], std::string{"cat"});
    EXPECT_EQ(labels[9], std::string{"truck"});

    const auto prediction = SoftmaxDecoder{}.decode(
        {10000.0F, 10001.0F, 9999.0F, 9998.0F, 9997.0F,
         9996.0F, 9995.0F, 9994.0F, 9993.0F, 9992.0F});
    EXPECT_EQ(prediction.class_index, std::size_t{1});
    EXPECT_EQ(prediction.label, std::string{"automobile"});
    EXPECT_EQ(prediction.probabilities.size(), std::size_t{10});
    EXPECT_TRUE(std::isfinite(prediction.confidence));

    float sum = 0.0F;
    for (const float probability : prediction.probabilities) {
        EXPECT_TRUE(std::isfinite(probability));
        EXPECT_TRUE(probability >= 0.0F && probability <= 1.0F);
        sum += probability;
    }
    EXPECT_NEAR(sum, 1.0F, kTolerance);
    EXPECT_NEAR(prediction.confidence, prediction.probabilities[1], kTolerance);
}

TEST_CASE(decoder_rejects_empty_mismatched_and_non_finite_logits) {
    EXPECT_THROW(SoftmaxDecoder{}.decode({}), std::invalid_argument);
    EXPECT_THROW(SoftmaxDecoder{}.decode({1.0F, 2.0F}), std::invalid_argument);

    const SoftmaxDecoder binary({"no", "yes"});
    EXPECT_THROW(
        binary.decode({0.0F, std::numeric_limits<float>::quiet_NaN()}),
        std::invalid_argument);
}

TEST_CASE(ppm_loader_reads_ascii_and_binary_rgb_without_external_dependencies) {
    TemporaryFile ascii_file(".ppm");
    {
        std::ofstream stream(ascii_file.path());
        stream << "P3\n# fixture comment\n2 1\n255\n255 0 1 2 3 4\n";
    }
    const Image ascii = cpp_ml::FileImageLoader{}.load(ascii_file.path());
    EXPECT_EQ(ascii.width, std::size_t{2});
    EXPECT_EQ(ascii.height, std::size_t{1});
    EXPECT_EQ(ascii.channels, std::size_t{3});
    EXPECT_EQ(ascii.pixels, (std::vector<std::uint8_t>{255, 0, 1, 2, 3, 4}));

    TemporaryFile binary_file(".ppm");
    {
        std::ofstream stream(binary_file.path(), std::ios::binary);
        stream << "P6\n1 1\n255\n";
        const std::array<char, 3> bytes = {char{0}, char{127}, static_cast<char>(255)};
        stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    const Image binary = cpp_ml::FileImageLoader{}.load(binary_file.path());
    EXPECT_EQ(binary.pixels, (std::vector<std::uint8_t>{0, 127, 255}));
}

TEST_CASE(ppm_binary_loader_handles_crlf_without_skipping_raster_whitespace) {
    TemporaryFile crlf_file(".ppm");
    {
        std::ofstream stream(crlf_file.path(), std::ios::binary);
        stream << "P6\r\n1 1\r\n255\r\n";
        const std::array<char, 3> bytes = {char{0}, char{127}, static_cast<char>(255)};
        stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    const Image crlf = cpp_ml::FileImageLoader{}.load(crlf_file.path());
    EXPECT_EQ(crlf.pixels, (std::vector<std::uint8_t>{0, 127, 255}));

    TemporaryFile whitespace_pixel(".ppm");
    {
        std::ofstream stream(whitespace_pixel.path(), std::ios::binary);
        stream << "P6\n1 1\n255\n";
        const std::array<char, 3> bytes = {char{10}, char{32}, char{13}};
        stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    const Image whitespace = cpp_ml::FileImageLoader{}.load(whitespace_pixel.path());
    EXPECT_EQ(whitespace.pixels, (std::vector<std::uint8_t>{10, 32, 13}));
}

TEST_CASE(ppm_ascii_loader_accepts_black_pixel_zero_regression) {
    TemporaryFile black_pixel(".ppm");
    {
        std::ofstream stream(black_pixel.path());
        stream << "P3\n1 1\n255\n0 0 0\n";
    }
    const Image image = cpp_ml::FileImageLoader{}.load(black_pixel.path());
    EXPECT_EQ(image.pixels, (std::vector<std::uint8_t>{0, 0, 0}));
}

TEST_CASE(ppm_loader_rejects_missing_or_unsupported_inputs) {
    EXPECT_THROW(
        cpp_ml::FileImageLoader{}.load("this-file-does-not-exist.ppm"),
        std::runtime_error);

    TemporaryFile unsupported(".ppm");
    {
        std::ofstream stream(unsupported.path());
        stream << "P2\n1 1\n255\n0\n";
    }
    EXPECT_THROW(cpp_ml::FileImageLoader{}.load(unsupported.path()), std::runtime_error);

    TemporaryFile truncated(".ppm");
    {
        std::ofstream stream(truncated.path(), std::ios::binary);
        stream << "P6\n2 2\n255\n";
        const std::array<char, 3> one_pixel = {char{1}, char{2}, char{3}};
        stream.write(one_pixel.data(), static_cast<std::streamsize>(one_pixel.size()));
    }
    EXPECT_THROW(cpp_ml::FileImageLoader{}.load(truncated.path()), std::runtime_error);

    TemporaryFile oversized(".ppm");
    {
        std::ofstream stream(oversized.path());
        stream << "P3\n18446744073709551615 18446744073709551615\n255\n0 0 0\n";
    }
    EXPECT_THROW(cpp_ml::FileImageLoader{}.load(oversized.path()), std::runtime_error);

    TemporaryFile long_token(".ppm");
    {
        std::ofstream stream(long_token.path());
        stream << "P3\n" << std::string(65, '9') << " 1\n255\n0 0 0\n";
    }
    EXPECT_THROW(cpp_ml::FileImageLoader{}.load(long_token.path()), std::runtime_error);
}

TEST_CASE(inference_engine_reuses_its_injected_backend) {
    auto state = std::make_shared<BackendState>();
    ModelOutput output = model_output(std::vector<float>(10, 1.0F), {1, 10}, 4.25);
    InferenceEngine engine(
        std::make_unique<RecordingBackend>(state, output));
    Tensor input = runtime_tensor(1);
    input.values.front() = 42.0F;

    const ModelOutput first = engine.infer(input);
    const ModelOutput second = engine.infer(input);
    EXPECT_EQ(state->calls, std::size_t{2});
    EXPECT_EQ(state->last_input.values, input.values);
    EXPECT_EQ(first.logits, output.logits);
    EXPECT_EQ(second.logits, output.logits);
    EXPECT_NEAR(first.inference_ms, 4.25, 1.0e-9);
}

TEST_CASE(inference_engine_rejects_invalid_input_and_backend_results) {
    EXPECT_THROW(InferenceEngine(nullptr), std::invalid_argument);

    const Tensor valid_input = runtime_tensor(1);

    auto input_state = std::make_shared<BackendState>();
    InferenceEngine input_engine(std::make_unique<RecordingBackend>(
        input_state, model_output(std::vector<float>(10, 1.0F), {1, 10})));
    EXPECT_THROW(input_engine.infer(Tensor{{1, 2}, {1.0F}}), std::invalid_argument);
    EXPECT_THROW(input_engine.infer(runtime_tensor(-1)), std::invalid_argument);
    EXPECT_THROW(input_engine.infer(runtime_tensor(0)), std::invalid_argument);
    EXPECT_THROW(input_engine.infer(Tensor{{257, 3, 32, 32}, {}}), std::invalid_argument);
    EXPECT_EQ(input_state->calls, std::size_t{0});

    auto empty_state = std::make_shared<BackendState>();
    InferenceEngine empty_engine(std::make_unique<RecordingBackend>(
        empty_state, model_output({}, {1, 10})));
    EXPECT_THROW(empty_engine.infer(valid_input), std::runtime_error);

    auto timing_state = std::make_shared<BackendState>();
    InferenceEngine timing_engine(std::make_unique<RecordingBackend>(
        timing_state, model_output(std::vector<float>(10, 1.0F), {1, 10}, -0.01)));
    EXPECT_THROW(timing_engine.infer(valid_input), std::runtime_error);

    auto infinite_state = std::make_shared<BackendState>();
    InferenceEngine infinite_engine(std::make_unique<RecordingBackend>(
        infinite_state,
        model_output(std::vector<float>(10, 1.0F), {1, 10},
                     std::numeric_limits<double>::infinity())));
    EXPECT_THROW(infinite_engine.infer(valid_input), std::runtime_error);

    auto nan_state = std::make_shared<BackendState>();
    InferenceEngine nan_engine(std::make_unique<RecordingBackend>(
        nan_state,
        model_output(std::vector<float>(10, 1.0F), {1, 10},
                     std::numeric_limits<double>::quiet_NaN())));
    EXPECT_THROW(nan_engine.infer(valid_input), std::runtime_error);

    auto logit_state = std::make_shared<BackendState>();
    auto non_finite_logits = std::vector<float>(10, 1.0F);
    non_finite_logits[4] = std::numeric_limits<float>::quiet_NaN();
    InferenceEngine logit_engine(std::make_unique<RecordingBackend>(
        logit_state, model_output(std::move(non_finite_logits), {1, 10})));
    EXPECT_THROW(logit_engine.infer(valid_input), std::runtime_error);

    auto shape_state = std::make_shared<BackendState>();
    InferenceEngine shape_engine(std::make_unique<RecordingBackend>(
        shape_state, model_output(std::vector<float>(9, 1.0F), {1, 9})));
    EXPECT_THROW(shape_engine.infer(valid_input), std::runtime_error);
}

TEST_CASE(inference_engine_preserves_valid_batch_shape_and_row_major_logits) {
    auto state = std::make_shared<BackendState>();
    std::vector<float> logits(80);
    for (std::size_t row = 0; row < 8; ++row) {
        for (std::size_t column = 0; column < 10; ++column) {
            logits[row * 10 + column] = static_cast<float>(row * 100 + column);
        }
    }
    InferenceEngine engine(std::make_unique<RecordingBackend>(
        state, model_output(logits, {8, 10}, 2.0)));

    const ModelOutput output = engine.infer(runtime_tensor(8));
    EXPECT_EQ(state->calls, std::size_t{1});
    EXPECT_EQ(output.shape, (std::vector<std::int64_t>{8, 10}));
    EXPECT_EQ(output.logits.size(), std::size_t{80});
    EXPECT_NEAR(output.logits[0], 0.0F, kTolerance);
    EXPECT_NEAR(output.logits[17], 107.0F, kTolerance);
    EXPECT_NEAR(output.logits[79], 709.0F, kTolerance);
}

TEST_CASE(pipeline_orchestrates_preprocessing_inference_and_decoding) {
    auto state = std::make_shared<BackendState>();
    ModelOutput output;
    output.logits = {0.0F, 0.0F, 0.0F, 8.0F, 0.0F,
                     0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    output.inference_ms = 0.0;
    output.shape = {1, 10};
    InferencePipeline pipeline(InferenceEngine(
        std::make_unique<RecordingBackend>(state, std::move(output))));

    const auto prediction = pipeline.predict(constant_image(32, 32, {0, 127, 255}));
    EXPECT_EQ(state->calls, std::size_t{1});
    EXPECT_EQ(state->last_input.shape, (std::vector<std::int64_t>{1, 3, 32, 32}));
    EXPECT_EQ(prediction.class_index, std::size_t{3});
    EXPECT_EQ(prediction.label, std::string{"cat"});
    EXPECT_NEAR(prediction.inference_ms, 0.0, 1.0e-9);
    EXPECT_TRUE(prediction.preprocessing_ms >= 0.0);
    EXPECT_TRUE(prediction.latency_ms >= 0.0);
}

}  // namespace

int main() {
    return test::run_all();
}
