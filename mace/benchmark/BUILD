# Benchmark
# Examples
load(
    "//mace:mace.bzl",
    "if_production_mode",
    "if_not_production_mode",
    "if_hexagon_enabled",
    "if_openmp_enabled",
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "stat_summarizer",
    srcs = ["stat_summarizer.cc"],
    hdrs = ["stat_summarizer.h"],
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
    linkopts = if_openmp_enabled(["-fopenmp"]),
    linkstatic = 1,
    deps = [
        ":stat_summarizer",
        "//external:gflags_nothreads",
        "//mace/codegen:generated_models",
    ],
)

cc_library(
    name = "libmace_merged",
    srcs = [
        "libmace_merged.a",
    ],
    visibility = ["//visibility:private"],
)

cc_binary(
    name = "model_throughput_test",
    srcs = ["model_throughput_test.cc"],
    linkopts = if_openmp_enabled(["-fopenmp"]),
    linkstatic = 1,
    deps = [
        ":libmace_merged",
        "//external:gflags_nothreads",
        "//mace/core",
    ],
)
