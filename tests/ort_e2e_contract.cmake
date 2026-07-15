foreach(required INFER_EXECUTABLE BENCHMARK_EXECUTABLE MODEL_PATH WORK_DIRECTORY)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

if(NOT EXISTS "${MODEL_PATH}")
    message(FATAL_ERROR "ONNX test model does not exist: ${MODEL_PATH}")
endif()

file(MAKE_DIRECTORY "${WORK_DIRECTORY}")
set(image_path "${WORK_DIRECTORY}/ort_e2e_black.ppm")
file(WRITE "${image_path}" "P3\n1 1\n255\n0 0 0\n")

execute_process(
    COMMAND "${INFER_EXECUTABLE}" --model "${MODEL_PATH}" --image "${image_path}"
    RESULT_VARIABLE infer_exit
    OUTPUT_VARIABLE infer_stdout
    ERROR_VARIABLE infer_stderr)
if(NOT infer_exit EQUAL 0)
    message(FATAL_ERROR
        "ORT CLI inference exited ${infer_exit}\n"
        "stdout:\n${infer_stdout}\nstderr:\n${infer_stderr}")
endif()
foreach(pattern
        "Prediction:[ 	]+[a-z]+"
        "Class index:[ 	]+[0-9]+"
        "Confidence:[ 	]+[0-9]+[.][0-9]+"
        "Latency:[ 	]+[0-9]+[.][0-9]+[ 	]+ms"
        "preprocess:[ 	]+[0-9]+[.][0-9]+[ 	]+ms"
        "inference:[ 	]+[0-9]+[.][0-9]+[ 	]+ms")
    if(NOT infer_stdout MATCHES "${pattern}")
        message(FATAL_ERROR
            "ORT CLI output did not match /${pattern}/\n${infer_stdout}")
    endif()
endforeach()

function(expect_cli_failure name pattern model image)
    execute_process(
        COMMAND "${INFER_EXECUTABLE}" --model "${model}" --image "${image}"
        RESULT_VARIABLE actual_exit
        OUTPUT_VARIABLE actual_stdout
        ERROR_VARIABLE actual_stderr)
    if(actual_exit EQUAL 0)
        message(FATAL_ERROR "${name}: invalid inference unexpectedly succeeded")
    endif()
    if(NOT actual_stderr MATCHES "${pattern}")
        message(FATAL_ERROR
            "${name}: stderr did not match /${pattern}/\n${actual_stderr}")
    endif()
endfunction()

expect_cli_failure(missing_model "not a regular file"
    "${WORK_DIRECTORY}/missing.onnx" "${image_path}")
expect_cli_failure(missing_image "could not open image"
    "${MODEL_PATH}" "${WORK_DIRECTORY}/missing.ppm")
set(invalid_image "${WORK_DIRECTORY}/ort_e2e_invalid.ppm")
file(WRITE "${invalid_image}" "P2\n1 1\n255\n0\n")
expect_cli_failure(unsupported_image "unsupported image format"
    "${MODEL_PATH}" "${invalid_image}")

execute_process(
    COMMAND "${BENCHMARK_EXECUTABLE}"
        --warmup 1 --iterations 2 --model "${MODEL_PATH}"
    RESULT_VARIABLE benchmark_exit
    OUTPUT_VARIABLE benchmark_stdout
    ERROR_VARIABLE benchmark_stderr)
if(NOT benchmark_exit EQUAL 0)
    message(FATAL_ERROR
        "ORT benchmark exited ${benchmark_exit}\n"
        "stdout:\n${benchmark_stdout}\nstderr:\n${benchmark_stderr}")
endif()
foreach(pattern
        "runtime_only"
        "end_to_end"
        "p50:[ 	]+[0-9]+[.][0-9]+[ 	]+ms"
        "p95:[ 	]+[0-9]+[.][0-9]+[ 	]+ms"
        "throughput:[ 	]+[0-9]+[.][0-9]+[ 	]+operations/s")
    if(NOT benchmark_stdout MATCHES "${pattern}")
        message(FATAL_ERROR
            "ORT benchmark output did not match /${pattern}/\n${benchmark_stdout}")
    endif()
endforeach()
