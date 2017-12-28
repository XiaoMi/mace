# Description:
# Mace operators.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android", "if_enable_neon")

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
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"] +
            if_enable_neon(["-DMACE_ENABLE_NEON"]),
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
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
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
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        ":ops",
        ":test",
        "//mace/core:test_benchmark_main",
    ],
)
