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
add_test(
    NAME cpp_ml_cli_contract
    COMMAND ${CMAKE_COMMAND}
        -DINFER_EXECUTABLE=$<TARGET_FILE:infer>
        -DWITH_ONNXRUNTIME=${WITH_ONNXRUNTIME}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cli_contract.cmake)

if(TARGET inference_benchmark)
    add_test(
        NAME cpp_ml_benchmark_contract
        COMMAND ${CMAKE_COMMAND}
            -DBENCHMARK_EXECUTABLE=$<TARGET_FILE:inference_benchmark>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_contract.cmake)
    set_tests_properties(cpp_ml_benchmark_contract PROPERTIES TIMEOUT 30)
endif()

set(CPP_ML_TEST_MODEL "" CACHE FILEPATH
    "Optional ONNX model used by the ONNX Runtime end-to-end CTest")
if(WITH_ONNXRUNTIME AND TARGET inference_benchmark AND
   CPP_ML_TEST_MODEL AND EXISTS "${CPP_ML_TEST_MODEL}")
    add_test(
        NAME cpp_ml_ort_e2e_contract
        COMMAND ${CMAKE_COMMAND}
            -DINFER_EXECUTABLE=$<TARGET_FILE:infer>
            -DBENCHMARK_EXECUTABLE=$<TARGET_FILE:inference_benchmark>
            -DMODEL_PATH=${CPP_ML_TEST_MODEL}
            -DWORK_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/ort_e2e_contract.cmake)
    set_tests_properties(cpp_ml_ort_e2e_contract PROPERTIES TIMEOUT 120)
endif()

find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME python_pipeline_smoke
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_python_pipeline.py)
    set_tests_properties(python_pipeline_smoke PROPERTIES TIMEOUT 120)
endif()

set_tests_properties(cpp_ml_core_tests cpp_ml_cli_contract PROPERTIES TIMEOUT 30)
