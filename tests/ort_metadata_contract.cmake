foreach(required INFER_EXECUTABLE FIXTURE_DIRECTORY WORK_DIRECTORY)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

file(MAKE_DIRECTORY "${WORK_DIRECTORY}")
set(image_path "${WORK_DIRECTORY}/ort_metadata_black.ppm")
file(WRITE "${image_path}" "P3\n1 1\n255\n0 0 0\n")

function(expect_model_rejected name pattern)
    set(model_path "${FIXTURE_DIRECTORY}/${name}.onnx")
    execute_process(
        COMMAND "${INFER_EXECUTABLE}" --model "${model_path}" --image "${image_path}"
        RESULT_VARIABLE actual_exit
        OUTPUT_VARIABLE actual_stdout
        ERROR_VARIABLE actual_stderr)
    if(actual_exit EQUAL 0)
        message(FATAL_ERROR "${name}: incompatible model unexpectedly succeeded")
    endif()
    if(NOT actual_stderr MATCHES "${pattern}")
        message(FATAL_ERROR
            "${name}: stderr did not match /${pattern}/\n${actual_stderr}")
    endif()
endfunction()

expect_model_rejected(wrong_input_name "endpoints must be named")
expect_model_rejected(wrong_output_name "endpoints must be named")
expect_model_rejected(wrong_input_dtype "input must have float32")
expect_model_rejected(wrong_input_rank "input must be NCHW")
expect_model_rejected(wrong_input_batch "batch")
expect_model_rejected(mixed_dynamic_input_fixed_output "batch")
expect_model_rejected(mixed_fixed_input_dynamic_output "batch")
expect_model_rejected(wrong_input_channels "input must be NCHW")
expect_model_rejected(wrong_input_height "input must be NCHW")
expect_model_rejected(wrong_input_width "input must be NCHW")
expect_model_rejected(dynamic_channel "input must be NCHW")
expect_model_rejected(dynamic_height "input must be NCHW")
expect_model_rejected(dynamic_width "input must be NCHW")
expect_model_rejected(extra_output "exactly one output")
expect_model_rejected(wrong_output_dtype "logits must have float32")
expect_model_rejected(wrong_output_shape "logits must have shape")
expect_model_rejected(wrong_output_rank "logits must have shape")
expect_model_rejected(dynamic_classes "logits must have shape")
expect_model_rejected(runtime_batch_mismatch "runtime shape")
expect_model_rejected(runtime_class_mismatch "runtime shape")
expect_model_rejected(corrupt "Inference failed")

execute_process(
    COMMAND "${INFER_EXECUTABLE}"
        --model "${FIXTURE_DIRECTORY}/missing.onnx" --image "${image_path}"
    RESULT_VARIABLE missing_exit
    ERROR_VARIABLE missing_stderr)
if(missing_exit EQUAL 0 OR NOT missing_stderr MATCHES "not a regular file")
    message(FATAL_ERROR "missing model did not fail with the expected diagnostic")
endif()
