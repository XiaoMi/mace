# Description:
# Mace operators.
#
package(
    default_visibility = ["//visibility:public"],
)


licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android")

cc_library(
    name = "ops",
    srcs = ["relu.cc"],
    hdrs = glob(["*.h"]),
    deps = [
        "//mace/proto:cc_proto",
        "//mace/core:core",
    ],
    copts = ['-std=c++11'],
    alwayslink = 1,
)

cc_test(
    name = "relu_test",
    srcs = ["relu_test.cc",],
    deps = [
        "@gtest//:gtest_main",
        ":ops",
    ],
    copts = ['-std=c++11'],
    linkopts = if_android([
        "-pie",
        "-llog",
        "-latomic",
        ]),
    linkstatic = 1,
)

