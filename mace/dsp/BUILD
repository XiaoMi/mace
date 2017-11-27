# Description:
# Mace dsp.
#
# Only suport arm-v7a now
# bazel build -c opt mace/dsp:dsp --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=armeabi-v7a --verbose_failures

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android")

cc_library(
    name = "dsp",
    srcs = glob([
        "*.cc",
        "hexagon/libhexagon_controller.so",
    ], exclude = [
        "*_test.cc",
    ]),
    hdrs = glob([
        "*.h",
        "hexagon/*.h",
    ]),
    copts = ["-std=c++11"],
    deps = [
        "//mace/proto:cc_proto",
        "//mace/core:core",
        "//mace/dsp/util:util",
    ],
)

cc_test(
    name = "dsp_test",
    testonly = 1,
    srcs = glob(["*_test.cc"]),
    copts = ["-std=c++11"],
    linkopts = if_android([
        "-ldl",
        "-lm",
    ]),
    linkstatic = 1,
    deps = [
        "@gtest//:gtest_main",
        ":dsp",
    ],
)

cc_test(
    name = "dsp_op_test",
    testonly = 1,
    srcs = glob(["test/*_test.cc"]),
    copts = ["-std=c++11"],
    linkopts = if_android([
        "-ldl",
        "-lm",
    ]),
    linkstatic = 1,
    deps = [
        "@gtest//:gtest_main",
        ":dsp",
        "//mace/kernels:kernels",
    ],
)
