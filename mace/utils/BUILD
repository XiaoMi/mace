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
    deps = [
        ":logging",
    ],
)

cc_library(
    name = "tuner",
    hdrs = [
        "timer.h",
        "tuner.h",
    ],
    deps = [
        ":logging",
        ":utils_hdrs",
    ],
)

cc_test(
    name = "tuner_test",
    testonly = 1,
    srcs = [
        "tuner_test.cc",
    ],
    linkopts = if_android([
        "-pie",
        "-lm",
    ]),
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
        "env_time.h",
        "utils.h",
    ],
)

cc_library(
    name = "utils",
    deps = [
        ":command_line_flags",
        ":logging",
        ":tuner",
        ":utils_hdrs",
    ],
)

cc_test(
    name = "utils_test",
    testonly = 1,
    srcs = [
        "utils_test.cc",
    ],
    linkopts = if_android([
        "-pie",
        "-lm",
    ]),
    linkstatic = 1,
    deps = [
        ":utils_hdrs",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)
