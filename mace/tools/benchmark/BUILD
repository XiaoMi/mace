# Benchmark
load("//mace:mace.bzl", "if_android")

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "stat_summarizer",
    srcs = ["stat_summarizer.cc"],
    hdrs = ["stat_summarizer.h"],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1"],
    linkstatic = 1,
    deps = [
        "//mace/core",
    ],
)

cc_binary(
    name = "benchmark_model",
    srcs = [
        "benchmark_model.cc",
    ],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1"],
    linkstatic = 1,
    deps = [
        ":stat_summarizer",
        "//mace/core",
        "//mace/ops",
        "//mace/utils:command_line_flags",
    ],
)
