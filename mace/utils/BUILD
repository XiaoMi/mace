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
        "logging.cc",
        "string_util.cc",
    ],
    hdrs = [
        "env_time.h",
        "logging.h",
        "memory_logging.h",
        "rwlock.h",
        "string_util.h",
        "timer.h",
        "tuner.h",
        "utils.h",
    ],
    linkopts = if_android([
        "-llog",
    ]),
    copts = ["-Werror"],
    deps = [
        "//mace/public",
        "//mace/proto:mace_cc",
    ],
)

cc_library(
    name = "utils_dev",
    srcs = [
        "tuner_development.cc",
    ],
    copts = ["-Werror"],
    deps = [
        ":utils",
    ],
)

cc_library(
    name = "utils_prod",
    srcs = [
        "tuner_production.cc",
    ],
    copts = ["-Werror"],
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
    copts = ["-Werror"],
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
