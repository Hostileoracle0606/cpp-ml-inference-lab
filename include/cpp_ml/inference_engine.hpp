#pragma once

#include "cpp_ml/domain.hpp"

#include <filesystem>
#include <memory>

namespace cpp_ml {

// Dependency inversion is intentionally limited to this volatile boundary.
// Domain values, preprocessing, and decoding remain concrete and lightweight.
class IInferenceBackend {
public:
    virtual ~IInferenceBackend() = default;
    [[nodiscard]] virtual ModelOutput run(const Tensor& input) = 0;
};

class InferenceEngine final {
public:
    explicit InferenceEngine(std::unique_ptr<IInferenceBackend> backend);
    ~InferenceEngine();

    InferenceEngine(InferenceEngine&&) noexcept;
    InferenceEngine& operator=(InferenceEngine&&) noexcept;
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    [[nodiscard]] ModelOutput infer(const Tensor& input);

private:
    std::unique_ptr<IInferenceBackend> backend_;
};

// Constructs a session-owning ONNX Runtime backend. The library is optional;
// when it was not enabled at build time this throws with an actionable message.
[[nodiscard]] std::unique_ptr<IInferenceBackend> make_onnxruntime_backend(
    const std::filesystem::path& model_path);

[[nodiscard]] bool onnxruntime_enabled() noexcept;

}  // namespace cpp_ml
