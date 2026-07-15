add_executable(cpp_ml_core_tests test_core.cpp)
target_link_libraries(cpp_ml_core_tests PRIVATE cpp_ml_core)
target_include_directories(cpp_ml_core_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(cpp_ml_core_tests PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(cpp_ml_core_tests PRIVATE /W4 /WX)
else()
    target_compile_options(cpp_ml_core_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_test(NAME cpp_ml_core_tests COMMAND cpp_ml_core_tests)
set_tests_properties(cpp_ml_core_tests PROPERTIES
    LABELS "cpp;portable;unit"
    TIMEOUT 30)

add_test(
    NAME cpp_ml_cli_contract
    COMMAND ${CMAKE_COMMAND}
        -DINFER_EXECUTABLE=$<TARGET_FILE:infer>
        -DWITH_ONNXRUNTIME=${WITH_ONNXRUNTIME}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cli_contract.cmake)
set_tests_properties(cpp_ml_cli_contract PROPERTIES
    LABELS "cli;integration;portable"
    TIMEOUT 30)

if(TARGET inference_benchmark)
    add_test(
        NAME cpp_ml_benchmark_contract
        COMMAND ${CMAKE_COMMAND}
            -DBENCHMARK_EXECUTABLE=$<TARGET_FILE:inference_benchmark>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_contract.cmake)
    set_tests_properties(cpp_ml_benchmark_contract PROPERTIES
        LABELS "benchmark;integration;portable"
        TIMEOUT 30)
endif()

option(CPP_ML_REQUIRE_PYTHON_DEPS
    "Fail configuration unless the complete Python release toolchain imports" OFF)
find_package(Python3 QUIET COMPONENTS Interpreter)
set(CPP_ML_PYTHON_FULL_AVAILABLE OFF)
if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
            "import torch, torchvision, onnx, onnxruntime, numpy, PIL"
        RESULT_VARIABLE CPP_ML_PYTHON_IMPORT_RESULT
        OUTPUT_QUIET
        ERROR_QUIET)
    if(CPP_ML_PYTHON_IMPORT_RESULT EQUAL 0)
        set(CPP_ML_PYTHON_FULL_AVAILABLE ON)
    endif()
endif()

if(CPP_ML_REQUIRE_PYTHON_DEPS AND NOT CPP_ML_PYTHON_FULL_AVAILABLE)
    message(FATAL_ERROR
        "CPP_ML_REQUIRE_PYTHON_DEPS=ON requires Python with torch, torchvision, "
        "onnx, onnxruntime, numpy, and Pillow. Set Python3_EXECUTABLE to the "
        "fully provisioned environment.")
endif()

if(CPP_ML_REQUIRE_PYTHON_DEPS)
    add_test(
        NAME python_dependency_import_gate
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/python_dependency_import_gate.py)
    set_tests_properties(python_dependency_import_gate PROPERTIES
        LABELS "python;release"
        TIMEOUT 30)
endif()

set(CPP_ML_TEST_MODEL "" CACHE FILEPATH
    "Optional ONNX model used by the ONNX Runtime end-to-end CTest")
if(WITH_ONNXRUNTIME AND TARGET inference_benchmark AND
   CPP_ML_PYTHON_FULL_AVAILABLE)
    set(CPP_ML_ONNX_FIXTURE_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/onnx_fixtures")
    add_test(
        NAME cpp_ml_generate_onnx_fixtures
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/generate_onnx_fixtures.py
            --output-dir ${CPP_ML_ONNX_FIXTURE_DIRECTORY})
    set_tests_properties(cpp_ml_generate_onnx_fixtures PROPERTIES
        FIXTURES_SETUP cpp_ml_onnx_fixtures
        LABELS "fixture;onnxruntime;python"
        TIMEOUT 60)

    add_test(
        NAME cpp_ml_ort_e2e_contract
        COMMAND ${CMAKE_COMMAND}
            -DINFER_EXECUTABLE=$<TARGET_FILE:infer>
            -DBENCHMARK_EXECUTABLE=$<TARGET_FILE:inference_benchmark>
            -DMODEL_PATH=${CPP_ML_ONNX_FIXTURE_DIRECTORY}/valid.onnx
            -DWORK_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/ort_e2e_contract.cmake)
    set_tests_properties(cpp_ml_ort_e2e_contract PROPERTIES
        FIXTURES_REQUIRED cpp_ml_onnx_fixtures
        LABELS "e2e;onnxruntime"
        TIMEOUT 120)

    add_test(
        NAME cpp_ml_ort_metadata_contract
        COMMAND ${CMAKE_COMMAND}
            -DINFER_EXECUTABLE=$<TARGET_FILE:infer>
            -DFIXTURE_DIRECTORY=${CPP_ML_ONNX_FIXTURE_DIRECTORY}
            -DWORK_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/ort_metadata_contract.cmake)
    set_tests_properties(cpp_ml_ort_metadata_contract PROPERTIES
        FIXTURES_REQUIRED cpp_ml_onnx_fixtures
        LABELS "integration;negative;onnxruntime"
        TIMEOUT 120)
endif()

if(WITH_ONNXRUNTIME AND TARGET inference_benchmark AND CPP_ML_TEST_MODEL)
    if(NOT EXISTS "${CPP_ML_TEST_MODEL}")
        message(FATAL_ERROR
            "CPP_ML_TEST_MODEL does not exist: ${CPP_ML_TEST_MODEL}")
    endif()
    add_test(
        NAME cpp_ml_ort_external_e2e_contract
        COMMAND ${CMAKE_COMMAND}
            -DINFER_EXECUTABLE=$<TARGET_FILE:infer>
            -DBENCHMARK_EXECUTABLE=$<TARGET_FILE:inference_benchmark>
            -DMODEL_PATH=${CPP_ML_TEST_MODEL}
            -DWORK_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/ort_e2e_contract.cmake)
    set_tests_properties(cpp_ml_ort_external_e2e_contract PROPERTIES
        LABELS "e2e;onnxruntime"
        TIMEOUT 120)
endif()

if(Python3_Interpreter_FOUND)
    add_test(
        NAME python_pipeline_smoke
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_python_pipeline.py)
    set_tests_properties(python_pipeline_smoke PROPERTIES
        LABELS "python;smoke"
        TIMEOUT 120)
endif()
