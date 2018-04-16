# Description:
# Mace neon kernels.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android", "if_neon_enabled", "if_openmp_enabled", "if_android_armv7")

cc_library(
    name = "kernels",
    srcs = glob(
        [
            "*.cc",
            "opencl/*.cc",
            "arm/*.cc",
        ],
        exclude = [
            "*_test.cc",
            "arm/*_test.cc",
            "opencl/*_test.cc",
        ],
    ),
    hdrs = glob([
        "*.h",
        "opencl/*.h",
        "arm/*.h",
    ]),
    copts = if_openmp_enabled(["-fopenmp"]) + if_neon_enabled(["-DMACE_ENABLE_NEON"]) + if_android_armv7(["-mfpu=neon -mfloat-abi=softfp"]),
    linkopts = if_android(["-lm"]),
    deps = [
        "//mace/core",
        "//mace/utils",
    ],
)

cc_test(
    name = "kernels_test",
    testonly = 1,
    srcs = glob(
        [
            "*_test.cc",
            "arm/*_test.cc",
            "opencl/*_test.cc",
        ],
    ),
    copts = if_openmp_enabled(["-fopenmp"]) + if_neon_enabled(["-DMACE_ENABLE_NEON"]),
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        ":kernels",
        "@gtest//:gtest",
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
