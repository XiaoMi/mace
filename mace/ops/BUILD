# Description:
# Mace operators.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android")

cc_library(
    name = "test",
    testonly = 1,
    hdrs = [
        "ops_test_util.h",
    ],
    deps = [
        "//mace/core",
        "@gtest//:gtest",
    ],
)

cc_library(
    name = "ops",
    srcs = glob(
        ["*.cc"],
        exclude = ["*_test.cc", "*_benchmark.cc"],
    ),
    hdrs = glob(
        ["*.h"],
        exclude = ["ops_test_util.h"],
    ),
    deps = [
        "//mace/core",
        "//mace/kernels",
        "//mace/proto:cc_proto",
    ],
    copts = ["-std=c++11"],
    alwayslink = 1,
)

cc_test(
    name = "ops_test",
    srcs = glob(
        ["*_test.cc"],
    ),
    deps = [
        ":ops",
        ":test",
        "@gtest//:gtest_main",
    ],
    copts = ["-std=c++11"],
    linkopts = if_android(["-ldl"]),
    linkstatic = 1,
    testonly = 1,
)

cc_test(
    name = "ops_benchmark",
    srcs = glob(["*_benchmark.cc"]),
    deps = [
        ":ops",
        ":test",
        "//mace/core:core",
        "//mace/core:test_benchmark_main",
    ],
    copts = ['-std=c++11'],
    linkopts = if_android(["-ldl"]),
    linkstatic = 1,
    testonly = 1,
)
