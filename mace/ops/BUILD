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
        "//mace/core:opencl_runtime",
        "@gtest//:gtest",
    ],
)

cc_library(
    name = "ops",
    srcs = glob(
        ["*.cc"],
        exclude = [
            "*_test.cc",
            "*_benchmark.cc",
        ],
    ),
    hdrs = glob(
        ["*.h"],
        exclude = ["ops_test_util.h"],
    ),
    copts = [
        "-std=c++11",
        "-fopenmp",
    ],
    deps = [
        "//mace/kernels",
        "//mace/proto:cc_proto",
    ],
    alwayslink = 1,
)

cc_test(
    name = "ops_test",
    testonly = 1,
    srcs = glob(
        ["*_test.cc"],
    ),
    copts = ["-std=c++11"],
    linkopts = ["-fopenmp"] + if_android(["-ldl"]),
    linkstatic = 1,
    deps = [
        ":ops",
        ":test",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "ops_benchmark",
    testonly = 1,
    srcs = glob(["*_benchmark.cc"]),
    copts = ["-std=c++11"],
    linkopts = ["-fopenmp"] + if_android(["-ldl"]),
    linkstatic = 1,
    deps = [
        ":ops",
        ":test",
        "//mace/core:test_benchmark_main",
    ],
)
