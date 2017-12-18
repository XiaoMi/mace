# Description:
# Mace dsp util.
#

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android")

cc_library(
    name = "util",
    srcs = glob([
        "*.cc",
    ], exclude = [
        "*_test.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1"],
    deps = [
        "//mace/core:core",
    ],
)

cc_test(
    name = "util_test",
    testonly = 1,
    srcs = glob(["*_test.cc"]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1"],
    linkopts = if_android([
        "-ldl",
        "-lm",
    ]),
    linkstatic = 1,
    deps = [
        "@gtest//:gtest_main",
        ":util",
    ],
)
