#include "cpp_ml/inference_engine.hpp"

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef CPP_ML_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace cpp_ml {

InferenceEngine::InferenceEngine(std::unique_ptr<IInferenceBackend> backend)
    : backend_(std::move(backend)) {
    if (!backend_) {
        throw std::invalid_argument("inference backend must not be null");
    }
}

InferenceEngine::~InferenceEngine() = default;
InferenceEngine::InferenceEngine(InferenceEngine&&) noexcept = default;
InferenceEngine& InferenceEngine::operator=(InferenceEngine&&) noexcept = default;

ModelOutput InferenceEngine::infer(const Tensor& input) {
    input.validate();
    auto output = backend_->run(input);
    if (output.logits.empty()) {
        throw std::runtime_error("inference backend returned no logits");
    }
    if (output.inference_ms < 0.0) {
        throw std::runtime_error("inference backend returned a negative duration");
    }
    return output;
}

#ifdef CPP_ML_WITH_ONNXRUNTIME
namespace {

class OnnxRuntimeBackend final : public IInferenceBackend {
public:
    explicit OnnxRuntimeBackend(const std::filesystem::path& model_path)
        : environment_(ORT_LOGGING_LEVEL_WARNING, "cpp_ml_inference"),
          session_options_{},
          session_{nullptr} {
        if (!std::filesystem::is_regular_file(model_path)) {
            throw std::invalid_argument("ONNX model is not a regular file: " +
                                        model_path.string());
        }
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session(environment_, model_path.c_str(), session_options_);

        if (session_.GetInputCount() != 1) {
            throw std::runtime_error("v1 expects an ONNX model with exactly one input");
        }
        if (session_.GetOutputCount() < 1) {
            throw std::runtime_error("ONNX model has no outputs");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        const auto input_name = session_.GetInputNameAllocated(0, allocator);
        const auto output_name = session_.GetOutputNameAllocated(0, allocator);
        input_name_ = input_name.get();
        output_name_ = output_name.get();
        if (input_name_ != "input" || output_name_ != "logits") {
            throw std::runtime_error(
                "ONNX endpoints must be named 'input' and 'logits'");
        }

        // TensorTypeAndShapeInfo is a non-owning view into TypeInfo. Keep the
        // owning TypeInfo alive while inspecting the view; chaining these
        // calls would leave input_info dangling at the end of the statement.
        const auto input_type = session_.GetInputTypeInfo(0);
        const auto input_info = input_type.GetTensorTypeAndShapeInfo();
        if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("ONNX input must have float32 element type");
        }
        const auto input_shape = input_info.GetShape();
        if (input_shape.size() != 4 ||
            (input_shape[0] != -1 && input_shape[0] != 1) ||
            (input_shape[1] != -1 && input_shape[1] != 3) ||
            (input_shape[2] != -1 && input_shape[2] != 32) ||
            (input_shape[3] != -1 && input_shape[3] != 32)) {
            throw std::runtime_error("ONNX input must be NCHW [N,3,32,32]");
        }

        const auto output_type = session_.GetOutputTypeInfo(0);
        const auto output_info = output_type.GetTensorTypeAndShapeInfo();
        if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("ONNX logits must have float32 element type");
        }
        const auto output_shape = output_info.GetShape();
        if (output_shape.size() != 2 ||
            (output_shape[0] != -1 && output_shape[0] != 1) ||
            (output_shape[1] != -1 && output_shape[1] != 10)) {
            throw std::runtime_error("ONNX logits must have shape [N,10]");
        }
    }

    ModelOutput run(const Tensor& input) override {
        if (input.shape.size() != 4 || input.shape[0] != 1 || input.shape[1] != 3 ||
            input.shape[2] != 32 || input.shape[3] != 32) {
            throw std::invalid_argument(
                "ONNX CIFAR-10 inference expects shape [1,3,32,32]");
        }

        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory, const_cast<float*>(input.values.data()), input.values.size(),
            input.shape.data(), input.shape.size());
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};

        const auto started = std::chrono::steady_clock::now();
        auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                    output_names, 1);
        const auto finished = std::chrono::steady_clock::now();
        if (outputs.size() != 1 || !outputs.front().IsTensor()) {
            throw std::runtime_error("ONNX Runtime returned an invalid output tensor");
        }
        const auto output_info = outputs.front().GetTensorTypeAndShapeInfo();
        if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("ONNX logits output must have float32 element type");
        }
        const auto output_shape = output_info.GetShape();
        if (output_shape != std::vector<std::int64_t>({1, 10})) {
            throw std::runtime_error("ONNX logits output must have runtime shape [1,10]");
        }
        const auto count = output_info.GetElementCount();
        if (count != 10) {
            throw std::runtime_error("ONNX logits output must contain 10 values for batch 1");
        }
        const float* values = outputs.front().GetTensorData<float>();

        ModelOutput result;
        result.logits.assign(values, values + count);
        result.inference_ms =
            std::chrono::duration<double, std::milli>(finished - started).count();
        return result;
    }

private:
    Ort::Env environment_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    std::string input_name_;
    std::string output_name_;
};

}  // namespace
#endif

std::unique_ptr<IInferenceBackend> make_onnxruntime_backend(
    const std::filesystem::path& model_path) {
#ifdef CPP_ML_WITH_ONNXRUNTIME
    return std::make_unique<OnnxRuntimeBackend>(model_path);
#else
    (void)model_path;
    throw std::runtime_error(
        "ONNX Runtime support is not enabled; configure with "
        "-DWITH_ONNXRUNTIME=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime");
#endif
}

bool onnxruntime_enabled() noexcept {
#ifdef CPP_ML_WITH_ONNXRUNTIME
    return true;
#else
    return false;
#endif
}

}  // namespace cpp_ml
