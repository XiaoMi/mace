# Description:
# Mace operators.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android", "if_neon_enabled", "if_openmp_enabled", "if_android_armv7")

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
        exclude = [
            "*_test.cc",
            "*_benchmark.cc",
        ],
    ),
    hdrs = glob(
        ["*.h"],
        exclude = ["ops_test_util.h"],
    ),
    copts = if_openmp_enabled(["-fopenmp"]) + if_neon_enabled(["-DMACE_ENABLE_NEON"]) + if_android_armv7(["-mfpu=neon -mfloat-abi=softfp"]),
    deps = [
        "//mace/kernels",
    ],
    alwayslink = 1,
)

cc_test(
    name = "ops_test",
    testonly = 1,
    srcs = glob(
        ["*_test.cc"],
    ),
    linkopts = ["-fopenmp"],
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
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        ":ops",
        ":test",
        "//mace/core:test_benchmark_main",
    ],
)
