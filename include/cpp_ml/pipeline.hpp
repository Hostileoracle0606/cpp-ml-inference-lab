#pragma once

#include "cpp_ml/decoder.hpp"
#include "cpp_ml/image_loader.hpp"
#include "cpp_ml/inference_engine.hpp"
#include "cpp_ml/preprocessor.hpp"

#include <filesystem>

namespace cpp_ml {

// Application facade that composes stable value-oriented components around the
// injected runtime backend. The backend/session is reused for every prediction.
class InferencePipeline final {
public:
    explicit InferencePipeline(InferenceEngine engine);

    [[nodiscard]] Prediction predict(const Image& image);
    [[nodiscard]] Prediction predict_file(const std::filesystem::path& path);

private:
    FileImageLoader loader_;
    Cifar10Preprocessor preprocessor_;
    SoftmaxDecoder decoder_;
    InferenceEngine engine_;
};

}  // namespace cpp_ml
