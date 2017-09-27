# Description:
# Mace neon kernels.
#
package(
    default_visibility = ["//visibility:public"],
)


licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android")

cc_library(
    name = "kernels",
    srcs = glob(["*.cc"]) + if_android(glob(["neon/*.cc"])),
    hdrs = glob(["*.h"]) + if_android(glob(["neon/*.h"])),
    deps = [
        "//mace/core:core",
    ],
    copts = ['-std=c++11', "-fopenmp",],
    linkopts = if_android(["-lm"]),
)

cc_test(
    name = "kernel_test",
    srcs = glob(["test/*.cc"]),
    deps = [
        "@gtest//:gtest_main",
        ":kernels",
        "//mace/core:core",
    ],
    copts = ['-std=c++11'],
    linkopts = if_android(["-pie"]),
    linkstatic = 1,
    testonly = 1,
)

cc_test(
    name = "benchmark",
    srcs = glob(["benchmark/*.cc"]),
    deps = [
        ":kernels",
        "//mace/core:core",
        "//mace/core:test_benchmark_main",
    ],
    copts = ['-std=c++11'],
    linkstatic = 1,
    testonly = 1,
)
