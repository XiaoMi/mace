# Description:
# Mace neon kernels.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android", "if_neon_enabled", "if_openmp_enabled")

cc_library(
    name = "kernels",
    srcs = glob([
        "*.cc",
        "opencl/*.cc",
    ]),
    hdrs = glob([
        "*.h",
        "opencl/*.h",
    ]),
    copts = if_openmp_enabled(["-fopenmp"]) + if_neon_enabled(["-DMACE_ENABLE_NEON"]),
    linkopts = if_android(["-lm"]),
    deps = [
        "//mace/core",
        "//mace/utils",
    ],
)

cc_test(
    name = "kernel_test",
    testonly = 1,
    srcs = glob(["test/*.cc"]),
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
    linkstatic = 1,
    deps = [
        ":kernels",
        "//mace/core",
        "//mace/core:test_benchmark_main",
    ],
)
