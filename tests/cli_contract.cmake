if(NOT DEFINED INFER_EXECUTABLE)
    message(FATAL_ERROR "INFER_EXECUTABLE is required")
endif()

function(run_cli_case name expected_exit stdout_pattern stderr_pattern)
    execute_process(
        COMMAND "${INFER_EXECUTABLE}" ${ARGN}
        RESULT_VARIABLE actual_exit
        OUTPUT_VARIABLE actual_stdout
        ERROR_VARIABLE actual_stderr)

    if(NOT actual_exit EQUAL expected_exit)
        message(FATAL_ERROR
            "${name}: expected exit ${expected_exit}, got ${actual_exit}\n"
            "stdout:\n${actual_stdout}\nstderr:\n${actual_stderr}")
    endif()
    if(NOT "${stdout_pattern}" STREQUAL "" AND
       NOT actual_stdout MATCHES "${stdout_pattern}")
        message(FATAL_ERROR
            "${name}: stdout did not match /${stdout_pattern}/\n${actual_stdout}")
    endif()
    if(NOT "${stderr_pattern}" STREQUAL "" AND
       NOT actual_stderr MATCHES "${stderr_pattern}")
        message(FATAL_ERROR
            "${name}: stderr did not match /${stderr_pattern}/\n${actual_stderr}")
    endif()
endfunction()

run_cli_case(help 0 "Usage: infer" "" --help)
run_cli_case(no_arguments 2 "" "Usage: infer")
run_cli_case(unknown_option 2 "" "Unknown.*--unknown" --unknown)
run_cli_case(missing_model_value 2 "" "[Mm]issing value.*--model" --model)
run_cli_case(missing_image_value 2 "" "[Mm]issing value.*--image" --image)
run_cli_case(model_without_image 2 "" "--image.*required" --model model.onnx)
run_cli_case(image_without_model 2 "" "--model.*required" --image image.ppm)
run_cli_case(duplicate_model 2 "" "more than once.*--model"
    --model first.onnx --model second.onnx --image image.ppm)

if(NOT WITH_ONNXRUNTIME)
    run_cli_case(runtime_disabled 1 "" "ONNX Runtime support is not enabled"
        --model model.onnx --image image.ppm)
endif()
