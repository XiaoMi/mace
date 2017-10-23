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
        "runtime/opencl/cl.hpp",
        "runtime/opencl/cl2.hpp",
        "runtime/opencl/*.cc",
    ]),
    hdrs = glob([
        "runtime/opencl/*.h",
    ]),
    copts = ["-std=c++11"],
    deps = [
        "core",
        "@opencl_headers//:opencl20_headers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "core",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    copts = ["-std=c++11"],
    linkopts = if_android([
        "-llog",
        "-pie",
    ]),
    deps = [
        "//mace/proto:cc_proto",
        "//mace/proto:stats_proto",
        "//mace/utils",
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
    copts = ["-std=c++11"],
    linkopts = ["-lm"],
    deps = [
        ":core",
    ],
    alwayslink = 1,
)
