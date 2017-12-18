# Description:
# Mace neon kernels.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android", "if_android_arm64")

cc_library(
    name = "kernels",
    srcs = glob(["*.cc", "opencl/*.cc"]) + if_android_arm64(glob([
        "neon/*.cc",
    ])),
    hdrs = glob(["*.h", "opencl/*.h"]) + if_android_arm64(glob([
        "neon/*.h",
    ])),
    copts = [
        "-std=c++11",
        "-fopenmp",
        "-D_GLIBCXX_USE_C99_MATH_TR1",
    ],
    linkopts = if_android(["-lm"]),
    deps = [
        "//mace/core",
        "//mace/core:opencl_runtime",
        "//mace/utils:utils_hdrs",
    ],
)

cc_test(
    name = "kernel_test",
    testonly = 1,
    srcs = glob(["test/*.cc"]),
    copts = ["-std=c++11"],
    linkopts = if_android(["-pie"]),
    linkstatic = 1,
    deps = [
        ":kernels",
        "//mace/core",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "benchmark",
    testonly = 1,
    srcs = glob(["benchmark/*.cc"]),
    copts = ["-std=c++11"],
    linkstatic = 1,
    deps = [
        ":kernels",
        "//mace/core",
        "//mace/core:test_benchmark_main",
    ],
)
