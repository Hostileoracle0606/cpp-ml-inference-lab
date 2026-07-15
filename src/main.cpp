#include "cpp_ml/inference_engine.hpp"
#include "cpp_ml/pipeline.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view kUsage =
    "Usage: infer --model <path.onnx> --image <path.ppm>\n"
    "\n"
    "Run CIFAR-10 inference for one RGB image. The portable v1 image loader\n"
    "accepts binary P6 and ASCII P3 PPM files.\n"
    "\n"
    "Required:\n"
    "  --model <path>   Exported CIFAR-10 ONNX model\n"
    "  --image <path>   RGB PPM input image\n"
    "\n"
    "Options:\n"
    "  -h, --help       Show this help and exit\n";

struct Arguments {
    std::string model;
    std::string image;
    bool help = false;
};

bool consume_value(int argc, char** argv, int& index, std::string& destination,
                   std::string_view option) {
    if (!destination.empty()) {
        std::cerr << "Argument provided more than once: " << option << '\n';
        return false;
    }
    if (index + 1 >= argc || std::string_view(argv[index + 1]).empty() ||
        std::string_view(argv[index + 1]).front() == '-') {
        std::cerr << "Missing value for argument: " << option << '\n';
        return false;
    }
    destination = argv[++index];
    return true;
}

bool parse_arguments(int argc, char** argv, Arguments& arguments) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "-h" || argument == "--help") {
            arguments.help = true;
        } else if (argument == "--model") {
            if (!consume_value(argc, argv, index, arguments.model, argument)) {
                return false;
            }
        } else if (argument == "--image") {
            if (!consume_value(argc, argv, index, arguments.image, argument)) {
                return false;
            }
        } else {
            std::cerr << "Unknown argument: " << argument << '\n';
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Arguments arguments;
    if (!parse_arguments(argc, argv, arguments)) {
        std::cerr << '\n' << kUsage;
        return 2;
    }
    if (arguments.help) {
        std::cout << kUsage;
        return 0;
    }
    if (arguments.model.empty() || arguments.image.empty()) {
        std::cerr << "Both --model and --image are required.\n\n" << kUsage;
        return 2;
    }

    try {
        cpp_ml::InferenceEngine engine(
            cpp_ml::make_onnxruntime_backend(arguments.model));
        cpp_ml::InferencePipeline pipeline(std::move(engine));
        const auto prediction = pipeline.predict_file(arguments.image);

        std::cout << std::fixed << std::setprecision(3)
                  << "Prediction: " << prediction.label << '\n'
                  << "Class index: " << prediction.class_index << '\n'
                  << "Confidence: " << prediction.confidence << '\n'
                  << "Latency:    " << prediction.latency_ms << " ms\n"
                  << "  preprocess: " << prediction.preprocessing_ms << " ms\n"
                  << "  inference:  " << prediction.inference_ms << " ms\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Inference failed: " << error.what() << '\n';
        return 1;
    }
}
