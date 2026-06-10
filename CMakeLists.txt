# ─────────────────────────────────────────────────────────────────────────────
# C++ ML Inference Lab — build definition
#
# Stage 0 builds a minimal, compilable CLI skeleton (`infer`) so the build is
# real from day one. ONNX Runtime linkage is gated behind WITH_ONNXRUNTIME and
# gets wired up in Stage 3; until then the project configures and builds without
# any external ML dependency installed.
# ─────────────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.16)

project(cpp_inference_lab
    VERSION 0.1.0
    DESCRIPTION "C++ ML inference system: PyTorch -> ONNX -> C++ runtime -> serving"
    LANGUAGES CXX)

# --- Language standard ---------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to a Release build for realistic performance numbers (Stage 5/6).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Options -------------------------------------------------------------------
option(WITH_ONNXRUNTIME "Link against ONNX Runtime for real inference (Stage 3+)" OFF)
option(BUILD_TESTS      "Build C++ unit tests (Stage 4+)"                          OFF)

# --- Main executable -----------------------------------------------------------
add_executable(infer src/main.cpp)
target_include_directories(infer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(infer PRIVATE -Wall -Wextra)

# --- Optional ONNX Runtime wiring (Stage 3) ------------------------------------
# Point CMake at your ONNX Runtime install with:
#   -DWITH_ONNXRUNTIME=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime
if(WITH_ONNXRUNTIME)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS ${ONNXRUNTIME_ROOT}/include
        PATH_SUFFIXES onnxruntime onnxruntime/core/session)
    find_library(ONNXRUNTIME_LIB onnxruntime
        HINTS ${ONNXRUNTIME_ROOT}/lib)

    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
        message(FATAL_ERROR
            "WITH_ONNXRUNTIME=ON but ONNX Runtime was not found. "
            "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime.")
    endif()

    target_compile_definitions(infer PRIVATE WITH_ONNXRUNTIME)
    target_include_directories(infer PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(infer PRIVATE ${ONNXRUNTIME_LIB})
    message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")
endif()

# --- Tests (Stage 4) -----------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    # add_subdirectory(tests)   # populated in Stage 4
endif()
