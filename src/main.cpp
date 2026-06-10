// C++ ML Inference Lab — CLI entry point.
//
// Stage 0: a compilable skeleton. It parses the --model / --image arguments and
// prints a plan, so `cmake --build` yields a working `infer` binary immediately.
// The real ONNX Runtime inference path is implemented in Stage 3 via the
// InferenceEngine class (include/inference_engine.hpp).

#include <iostream>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view kUsage =
    "Usage: infer --model <path.onnx> --image <path.png>\n"
    "\n"
    "Options:\n"
    "  --model <path>   Path to the exported ONNX model (models/cifar10_cnn.onnx)\n"
    "  --image <path>   Path to an input image (samples/cat.png)\n"
    "  -h, --help       Show this help and exit\n";

struct Args {
    std::string model;
    std::string image;
    bool help = false;
};

// Minimal hand-rolled parser — no external dependency for a two-flag CLI.
// Returns false on a malformed flag (missing value / unknown option).
bool parse_args(int argc, char** argv, Args& out) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            out.help = true;
        } else if (arg == "--model" && i + 1 < argc) {
            out.model = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            out.image = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        std::cerr << "\n" << kUsage;
        return 2;
    }
    if (args.help || (args.model.empty() && args.image.empty())) {
        std::cout << kUsage;
        return 0;
    }

    std::cout << "C++ ML Inference Lab\n"
              << "  model: " << (args.model.empty() ? "(none)" : args.model) << "\n"
              << "  image: " << (args.image.empty() ? "(none)" : args.image) << "\n";

#ifdef WITH_ONNXRUNTIME
    std::cout << "  build: ONNX Runtime linked\n";
#else
    std::cout << "  build: skeleton (configure with -DWITH_ONNXRUNTIME=ON for Stage 3)\n";
#endif

    std::cout << "\n[Stage 0] Inference is not wired up yet.\n"
              << "Next: implement InferenceEngine::predict() in Stage 3 to load the\n"
              << "ONNX model, preprocess the image, run the session, and print:\n"
              << "  Prediction: <label>\n  Confidence: <0..1>\n  Latency:    <ms>\n";
    return 0;
}
