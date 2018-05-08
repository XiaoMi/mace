# Benchmark
# Examples
load(
    "//mace:mace.bzl",
    "if_production_mode",
    "if_not_production_mode",
    "if_hexagon_enabled",
    "if_openmp_enabled",
    "if_android",
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "statistics",
    srcs = ["statistics.cc"],
    hdrs = ["statistics.h"],
    linkstatic = 1,
    deps = [
        "//mace/kernels",
        "//mace/utils",
    ],
)

cc_binary(
    name = "benchmark_model",
    srcs = [
        "benchmark_model.cc",
    ],
    copts = if_android(["-DMACE_ENABLE_OPENCL"]),
    linkopts = if_openmp_enabled(["-fopenmp"]),
    linkstatic = 1,
    deps = [
        ":statistics",
        "//external:gflags_nothreads",
        "//mace/codegen:generated_models",
        "//mace/codegen:generated_mace_engine_creator",
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
