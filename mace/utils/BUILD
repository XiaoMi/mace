# Description:
# Mace utils.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android")

cc_library(
    name = "logging",
    srcs = [
        "logging.cc",
    ],
    hdrs = [
        "logging.h",
    ],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    linkopts = if_android([
        "-llog",
    ]),
)

cc_library(
    name = "command_line_flags",
    srcs = [
        "command_line_flags.cc",
    ],
    hdrs = [
        "command_line_flags.h",
    ],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    deps = [
        ":logging",
    ],
)

cc_library(
    name = "tuner",
    hdrs = [
        "tuner.h",
        "timer.h",
    ],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    deps = [
        ":utils_hdrs",
        ":logging",
    ],
)

cc_test(
    name = "tuner_test",
    testonly = 1,
    srcs = [
        "tuner_test.cc",
    ],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    linkopts = if_android(["-pie", "-lm"]),
    linkstatic = 1,
    deps = [
        ":tuner",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "utils_hdrs",
    hdrs = [
        "utils.h",
        "env_time.h",
    ],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
)

cc_library(
    name = "utils",
    deps = [
        ":utils_hdrs",
        ":tuner",
        ":command_line_flags",
        ":logging",
    ],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
)
