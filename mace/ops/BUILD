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
        exclude = ["*_test.cc"],
    ),
    hdrs = glob(
        ["*.h"],
        exclude = ["ops_test_util.h"],
    ),
    copts = ["-std=c++11"],
    deps = [
        "//mace/core",
        "//mace/kernels",
        "//mace/proto:cc_proto",
    ],
    alwayslink = 1,
)

cc_test(
    name = "relu_test",
    srcs = ["relu_test.cc"],
    copts = ["-std=c++11"],
    linkopts = if_android([
        "-pie",
        "-llog",
        "-latomic",
    ]),
    linkstatic = 1,
    deps = [
        ":ops",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "pooling_test",
    srcs = ["pooling_test.cc"],
    copts = ["-std=c++11"],
    linkopts = if_android([
        "-pie",
        "-llog",
        "-latomic",
    ]),
    linkstatic = 1,
    deps = [
        ":ops",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "batch_norm_test",
    srcs = ["batch_norm_test.cc"],
    copts = ["-std=c++11"],
    linkstatic = 1,
    deps = [
        ":ops",
        ":test",
        "@gtest//:gtest_main",
    ],
)
