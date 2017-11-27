# Description:
# Mace core.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android")

cc_library(
    name = "opencl_runtime",
    srcs = glob([
        "runtime/opencl/*.cc",
    ]),
    hdrs = glob([
        "runtime/opencl/cl.hpp",
        "runtime/opencl/cl2.hpp",
        "runtime/opencl/*.h",
    ]),
    copts = ["-std=c++11"],
    deps = [
        ":logging",
        "@opencl_headers//:opencl20_headers",
    ],
    alwayslink = 1,
)


cc_library(
    name = "logging",
    srcs = [
        "logging.cc",
    ],
    hdrs = [
        "logging.h",
    ],
    copts = ["-std=c++11"],
    linkopts = if_android([
        "-llog",
    ]),
)

cc_library(
    name = "core",
    srcs = glob(
        ["*.cc",],
        exclude=[
            "logging.cc",
        ]),
    hdrs = glob(
        ["*.h"],
        exclude=[
            "logging.h",
        ]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1"] + if_android([
            "-D__USE_OPENCL",
    ]),
    linkopts = ["-ldl"] + if_android([
        "-pie",
    ]),
    deps = [
        ":logging",
        "//mace/proto:cc_proto",
        "//mace/proto:stats_proto",
        "//mace/utils",
        ":opencl_runtime",
    ],
)

# Main program for tests
cc_library(
    name = "test_benchmark_main",
    testonly = 1,
    srcs = glob([
        "testing/*.cc",
    ]),
    hdrs = glob([
        "testing/*.h",
    ]),
    copts = [
        "-std=c++11",
        "-D_GLIBCXX_USE_C99_MATH_TR1",
    ],
    linkopts = ["-lm"],
    deps = [
        ":core",
    ],
    alwayslink = 1,
)
