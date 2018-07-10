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
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
    deps = [
        "//mace/public",
    ],
)

cc_test(
    name = "tuner_test",
    testonly = 1,
    srcs = [
        "tuner_test.cc",
    ],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
    linkopts = ["-ldl"] + if_android([
        "-pie",
        "-lm",  # Required by unordered_map
    ]),
    linkstatic = 1,
    deps = [
        ":utils",
        "//mace/core",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)
