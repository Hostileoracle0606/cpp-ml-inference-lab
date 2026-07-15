if(NOT DEFINED BENCHMARK_EXECUTABLE)
    message(FATAL_ERROR "BENCHMARK_EXECUTABLE is required")
endif()

execute_process(
    COMMAND "${BENCHMARK_EXECUTABLE}" --warmup 1 --iterations 3
    RESULT_VARIABLE benchmark_exit
    OUTPUT_VARIABLE benchmark_stdout
    ERROR_VARIABLE benchmark_stderr)

if(NOT benchmark_exit EQUAL 0)
    message(FATAL_ERROR
        "offline benchmark exited ${benchmark_exit}\n"
        "stdout:\n${benchmark_stdout}\nstderr:\n${benchmark_stderr}")
endif()

foreach(pattern
        "build[_ ]type:[ 	]+[A-Za-z]+"
        "system:[ 	]+[A-Za-z0-9]"
        "compiler:[ 	]+[A-Za-z0-9]"
        "warmup:[ 	]+1"
        "iterations:[ 	]+3"
        "preprocessing"
        "mean:[ 	]+[0-9]+[.][0-9]+[ 	]+ms"
        "throughput:[ 	]+[0-9]+[.][0-9]+[ 	]+operations/s"
        "inference:[ 	]+skipped")
    if(NOT benchmark_stdout MATCHES "${pattern}")
        message(FATAL_ERROR
            "offline benchmark output did not match /${pattern}/\n${benchmark_stdout}")
    endif()
endforeach()

if(NOT benchmark_stdout MATCHES "p50:[ 	]+([0-9]+[.][0-9]+)[ 	]+ms")
    message(FATAL_ERROR "benchmark output has no finite p50\n${benchmark_stdout}")
endif()
set(p50 "${CMAKE_MATCH_1}")
if(NOT benchmark_stdout MATCHES "p95:[ 	]+([0-9]+[.][0-9]+)[ 	]+ms")
    message(FATAL_ERROR "benchmark output has no finite p95\n${benchmark_stdout}")
endif()
set(p95 "${CMAKE_MATCH_1}")
if(p95 LESS p50)
    message(FATAL_ERROR "benchmark p95 (${p95}) must be >= p50 (${p50})")
endif()

function(expect_benchmark_error name pattern)
    execute_process(
        COMMAND "${BENCHMARK_EXECUTABLE}" ${ARGN}
        RESULT_VARIABLE actual_exit
        OUTPUT_VARIABLE actual_stdout
        ERROR_VARIABLE actual_stderr)
    if(actual_exit EQUAL 0)
        message(FATAL_ERROR "${name}: invalid invocation unexpectedly succeeded")
    endif()
    if(NOT actual_stderr MATCHES "${pattern}")
        message(FATAL_ERROR
            "${name}: stderr did not match /${pattern}/\n${actual_stderr}")
    endif()
endfunction()

expect_benchmark_error(zero_iterations "invalid value.*--iterations"
    --iterations 0)
expect_benchmark_error(non_numeric_warmup "invalid value.*--warmup"
    --warmup abc)
expect_benchmark_error(unknown_argument "unknown argument.*--unknown"
    --unknown 1)
expect_benchmark_error(missing_value "missing value.*--iterations"
    --iterations)
