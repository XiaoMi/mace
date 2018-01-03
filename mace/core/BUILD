# Description:
# Mace core.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android", "if_profiling_enabled")

cc_library(
    name = "core",
    srcs = glob([
        "*.cc",
        "runtime/opencl/*.cc",
    ], exclude=["runtime/opencl/opencl_binary_linkage.cc",
        "runtime/opencl/opencl_source_linkage.cc",
    ]),
    hdrs = glob([
        "*.h",
        "public/*.h",
        "runtime/opencl/cl2.hpp",
        "runtime/opencl/*.h",
    ]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"] +
            if_profiling_enabled(["-DMACE_OPENCL_PROFILING"]),
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

cc_library(
    name = "opencl_source_linkage",
    srcs = ["runtime/opencl/opencl_source_linkage.cc"],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    linkstatic = 1,
    deps = [
        ":core",
    ],
)

cc_library(
    name = "opencl_binary_linkage",
    srcs = ["runtime/opencl/opencl_binary_linkage.cc"],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    linkstatic = 1,
    deps = [
        ":core",
    ],
)
