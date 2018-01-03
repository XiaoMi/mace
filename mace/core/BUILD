# Description:
# Mace core.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android", "if_android_armv7", "if_android_arm64",

 "if_profiling_enabled", "if_embed_binary_program")

cc_library(
    name = "core",
    srcs = glob([
        "*.cc",
        "runtime/opencl/*.cc",
        "runtime/hexagon/*.cc",
    ],
    exclude = [
        "*_test.cc",
        "runtime/hexagon/hexagon_controller_dummy.cc",
    ]) + if_android_armv7([
        "runtime/hexagon/libhexagon_controller.so",
    ]) + if_android_arm64([
        "runtime/hexagon/hexagon_controller_dummy.cc",
    ]),
    hdrs = glob([
        "*.h",
        "public/*.h",
        "runtime/opencl/cl2.hpp",
        "runtime/opencl/*.h",
        "runtime/hexagon/*.h",
    ]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"] +
            if_profiling_enabled(["-DMACE_OPENCL_PROFILING"]) +
            if_embed_binary_program(["-DMACE_EMBED_BINARY_PROGRAM"]),
    linkopts = if_android(["-pie", "-ldl"]),
    deps = [
        "//mace/utils:utils_hdrs",
        "//mace/utils:logging",
        "//mace/utils:tuner",
        "@opencl_headers//:opencl20_headers",
    ],
)

cc_library(
    name = "test_benchmark_main",
    testonly = 1,
    hdrs = [
        "testing/test_benchmark.h",
    ],
    srcs = [
        "testing/test_benchmark.cc",
        "testing/test_benchmark_main.cc",
    ],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    alwayslink = 1,
    deps = [
        ":core",
        "//mace/utils:utils_hdrs",
    ],
)
