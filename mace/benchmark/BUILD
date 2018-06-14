# Benchmark
# Examples
load(
    "//mace:mace.bzl",
    "if_hexagon_enabled",
    "if_openmp_enabled",
    "if_android",
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "statistics",
    srcs = ["statistics.cc"],
    hdrs = ["statistics.h"],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
    deps = [
        "//mace/utils",
    ],
)

cc_binary(
    name = "benchmark_model_static",
    srcs = [
        "benchmark_model.cc",
    ],
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + if_android(["-DMACE_ENABLE_OPENCL"]),
    linkopts = if_openmp_enabled(["-fopenmp"]),
    linkstatic = 1,
    deps = [
        ":statistics",
        "//external:gflags_nothreads",
        "//mace/codegen:generated_models",
        "//mace/codegen:generated_mace_engine_factory",
    ],
)

cc_binary(
    name = "benchmark_model_shared",
    srcs = [
        "benchmark_model.cc",
    ],
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + if_android(["-DMACE_ENABLE_OPENCL"]),
    linkopts = ["-lm", "-pie", "-fPIE"] + if_openmp_enabled(["-fopenmp"]),
    linkstatic = 0,
    deps = [
        ":statistics",
        "//external:gflags_nothreads",
        "//mace/codegen:generated_mace_engine_factory",
        "//mace:libmace",
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
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
    deps = [
        ":libmace_merged",
        "//external:gflags_nothreads",
        "//mace/core",
    ],
)
