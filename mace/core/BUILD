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
        "platform/opencl/cl.hpp",
        "platform/opencl/cl2.hpp",
        "platform/opencl/opencl_wrapper.h",
        "platform/opencl/opencl_wrapper.cc",
    ]),
    copts = ["-std=c++11"],
    deps = [
        "@opencl_headers//:opencl12_headers",
        "core",
    ],
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
    deps = [
        "//mace/proto:cc_proto",
    ],
    linkopts = if_android([
        "-llog",
        "-pie",
    ]),
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
