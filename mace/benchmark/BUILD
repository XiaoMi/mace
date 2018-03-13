# Benchmark
# Examples
load(
    "//:mace.bzl",
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
        "@mace//:mace_headers",
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
        "//mace/codegen:generated_models",
        "//external:gflags_nothreads",
    ] + if_hexagon_enabled([
        "//lib/hexagon:hexagon",
    ]) + if_production_mode([
        "@mace//:mace_prod",
        "//codegen:generated_opencl_prod",
        "//codegen:generated_tuning_params",
    ]) + if_not_production_mode([
        "@mace//:mace_dev",
    ]),
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
        "//lib/hexagon",
        "@mace//:mace",
        "@mace//:mace_headers",
        "@mace//:mace_prod",
    ],
)
