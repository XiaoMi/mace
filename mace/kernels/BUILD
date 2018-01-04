# Description:
# Mace neon kernels.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android", "if_enable_neon")

cc_library(
    name = "kernels",
    srcs = glob(["*.cc", "opencl/*.cc"]) + if_enable_neon(glob([
        "neon/*.cc",
    ])),
    hdrs = glob(["*.h", "opencl/*.h"]) + if_enable_neon(glob([
        "neon/*.h",
    ])),
    copts = [
        "-std=c++11",
        "-D_GLIBCXX_USE_C99_MATH_TR1",
        "-Werror=return-type",
    ] +
    if_enable_neon(["-fopenmp"]),
    linkopts = if_android(["-lm"]),
    deps = [
        "//mace/core",
        "//mace/utils:utils_hdrs",
    ],
)

cc_test(
    name = "kernel_test",
    testonly = 1,
    srcs = glob(["test/*.cc"]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
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
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    linkstatic = 1,
    deps = [
        ":kernels",
        "//mace/core",
        "//mace/core:test_benchmark_main",
    ],
)
