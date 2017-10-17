# Benchmark
load("//mace:mace.bzl", "if_android")

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "stat_summarizer",
    srcs = ["stat_summarizer.cc"],
    hdrs = ["stat_summarizer.h"],
    copts = ["-std=c++11"],
    linkopts = ["-fopenmp"] + if_android([
        "-ldl",
        "-lm",
    ]),
    linkstatic = 1,
    deps = [
        "//mace/core",
        "//mace/proto:stats_proto",
    ],
)

cc_binary(
    name = "benchmark_model",
    srcs = [
        "benchmark_model.cc",
    ],
    copts = ["-std=c++11"],
    linkopts = ["-fopenmp"] + if_android(["-ldl"]),
    linkstatic = 1,
    deps = [
        ":stat_summarizer",
        "//mace/core",
        "//mace/ops",
        "//mace/utils:command_line_flags",
    ],
)
