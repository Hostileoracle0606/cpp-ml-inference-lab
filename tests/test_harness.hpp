#pragma once

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace test {

using Function = std::function<void()>;

struct Case {
    std::string name;
    Function function;
};

inline std::vector<Case>& cases() {
    static std::vector<Case> registered;
    return registered;
}

struct Registrar {
    Registrar(std::string name, Function function) {
        cases().push_back({std::move(name), std::move(function)});
    }
};

[[noreturn]] inline void fail(
    std::string_view expression,
    std::string_view file,
    int line,
    std::string_view detail = {}) {
    std::ostringstream message;
    message << file << ':' << line << ": assertion failed: " << expression;
    if (!detail.empty()) {
        message << " (" << detail << ')';
    }
    throw std::runtime_error(message.str());
}

template <typename Exception, typename FunctionType>
void expect_throw(FunctionType&& function, std::string_view file, int line) {
    try {
        std::forward<FunctionType>(function)();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        fail("expected exception type", file, line, error.what());
    }
    fail("expected exception", file, line);
}

inline int run_all() {
    std::size_t failures = 0;
    for (const auto& test_case : cases()) {
        try {
            test_case.function();
            std::cout << "[PASS] " << test_case.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test_case.name << ": " << error.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << test_case.name << ": unknown exception\n";
        }
    }
    std::cout << (cases().size() - failures) << '/' << cases().size()
              << " tests passed\n";
    return failures == 0 ? 0 : 1;
}

}  // namespace test

#define TEST_CASE(name)                                                        \
    static void name();                                                        \
    static const test::Registrar name##_registrar{#name, name};               \
    static void name()

#define EXPECT_TRUE(expression)                                                \
    do {                                                                       \
        if (!(expression)) {                                                   \
            test::fail(#expression, __FILE__, __LINE__);                       \
        }                                                                      \
    } while (false)

#define EXPECT_EQ(actual, expected)                                            \
    do {                                                                       \
        const auto actual_value = (actual);                                    \
        const auto expected_value = (expected);                                \
        if (!(actual_value == expected_value)) {                               \
            test::fail(#actual " == " #expected, __FILE__, __LINE__);          \
        }                                                                      \
    } while (false)

#define EXPECT_NEAR(actual, expected, tolerance)                               \
    do {                                                                       \
        const auto actual_value = static_cast<double>(actual);                 \
        const auto expected_value = static_cast<double>(expected);             \
        const auto tolerance_value = static_cast<double>(tolerance);           \
        if (!std::isfinite(actual_value) ||                                    \
            std::abs(actual_value - expected_value) > tolerance_value) {       \
            test::fail(#actual " ~= " #expected, __FILE__, __LINE__);          \
        }                                                                      \
    } while (false)

#define EXPECT_THROW(statement, exception_type)                                \
    test::expect_throw<exception_type>(                                         \
        [&] { static_cast<void>(statement); }, __FILE__, __LINE__)
