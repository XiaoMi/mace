# Description:
# Mace utils.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android")

cc_library(
    name = "utils",
    srcs = [
        "command_line_flags.cc",
        "logging.cc",
    ],
    hdrs = [
        "command_line_flags.h",
        "env_time.h",
        "logging.h",
        "memory_logging.h",
        "string_util.h",
        "timer.h",
        "tuner.h",
        "utils.h",
        "rwlock.h",
    ],
    linkopts = if_android([
        "-llog",
    ]),
    deps = [
        "//mace/public",
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
        ":utils",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "utils_dev",
    srcs = [
        "tuner_development.cc",
    ],
    deps = [
        ":utils",
    ],
)

cc_library(
    name = "utils_prod",
    srcs = [
        "tuner_production.cc",
    ],
    deps = [
        ":utils",
        "//mace/codegen:generated_tuning_params",
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
        "-lm",  # Required by unordered_map
    ]),
    linkstatic = 1,
    deps = [
        ":utils_dev",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)
