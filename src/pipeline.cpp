#include "cpp_ml/pipeline.hpp"

#include <chrono>
#include <utility>

namespace cpp_ml {

InferencePipeline::InferencePipeline(InferenceEngine engine)
    : engine_(std::move(engine)) {}

Prediction InferencePipeline::predict(const Image& image) {
    const auto started = std::chrono::steady_clock::now();
    const auto preprocessed = preprocessor_.preprocess(image);
    const auto preprocessing_finished = std::chrono::steady_clock::now();
    auto model_output = engine_.infer(preprocessed);
    auto prediction = decoder_.decode(model_output.logits);
    const auto finished = std::chrono::steady_clock::now();

    prediction.preprocessing_ms =
        std::chrono::duration<double, std::milli>(preprocessing_finished - started)
            .count();
    prediction.inference_ms = model_output.inference_ms;
    prediction.latency_ms =
        std::chrono::duration<double, std::milli>(finished - started).count();
    return prediction;
}

Prediction InferencePipeline::predict_file(const std::filesystem::path& path) {
    const auto started = std::chrono::steady_clock::now();
    const auto image = loader_.load(path);
    auto prediction = predict(image);
    const auto finished = std::chrono::steady_clock::now();
    prediction.latency_ms =
        std::chrono::duration<double, std::milli>(finished - started).count();
    return prediction;
}

}  // namespace cpp_ml
