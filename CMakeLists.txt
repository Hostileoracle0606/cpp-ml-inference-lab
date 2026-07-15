cmake_minimum_required(VERSION 3.16)

project(cpp_inference_lab
    VERSION 1.1.0
    DESCRIPTION "Portable C++17 CIFAR-10 inference pipeline"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(WITH_ONNXRUNTIME "Build the optional ONNX Runtime backend" OFF)
option(BUILD_BENCHMARKS "Build the inference benchmark harness" ON)

# Preserve the Stage 0 BUILD_TESTS spelling without letting it fight CTest's
# standard BUILD_TESTING option. A caller that supplies the legacy cache entry
# controls BUILD_TESTING; otherwise CTest provides its normal default.
if(DEFINED BUILD_TESTS)
    set(BUILD_TESTING "${BUILD_TESTS}" CACHE BOOL "Build tests" FORCE)
endif()
include(CTest)

function(cpp_ml_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_library(cpp_ml_core
    src/decoder.cpp
    src/domain.cpp
    src/image_loader.cpp
    src/inference_engine.cpp
    src/pipeline.cpp
    src/preprocessor.cpp)
target_include_directories(cpp_ml_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_features(cpp_ml_core PUBLIC cxx_std_17)

cpp_ml_enable_warnings(cpp_ml_core)

if(WITH_ONNXRUNTIME)
    set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of an ONNX Runtime release")
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}/include"
        PATH_SUFFIXES onnxruntime onnxruntime/core/session)
    find_library(ONNXRUNTIME_LIBRARY onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}/lib")
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR
            "WITH_ONNXRUNTIME=ON but ONNX Runtime was not found. "
            "Set ONNXRUNTIME_ROOT to an unpacked release directory.")
    endif()
    target_compile_definitions(cpp_ml_core PRIVATE CPP_ML_WITH_ONNXRUNTIME=1)
    target_include_directories(cpp_ml_core PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(cpp_ml_core PRIVATE "${ONNXRUNTIME_LIBRARY}")
    message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
endif()

add_executable(infer src/main.cpp)
target_link_libraries(infer PRIVATE cpp_ml_core)
cpp_ml_enable_warnings(infer)

if(BUILD_BENCHMARKS)
    add_executable(inference_benchmark benchmarks/inference_benchmark.cpp)
    target_link_libraries(inference_benchmark PRIVATE cpp_ml_core)
    cpp_ml_enable_warnings(inference_benchmark)
    target_compile_definitions(inference_benchmark
        PRIVATE
            CPP_ML_BUILD_TYPE="$<CONFIG>"
            CPP_ML_SYSTEM_NAME="${CMAKE_SYSTEM_NAME}"
            CPP_ML_SYSTEM_PROCESSOR="${CMAKE_SYSTEM_PROCESSOR}"
            CPP_ML_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
            CPP_ML_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}")
endif()

if(BUILD_TESTING)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        message(FATAL_ERROR
            "BUILD_TESTING=ON but tests/CMakeLists.txt is missing")
    endif()
    add_subdirectory(tests)
endif()
