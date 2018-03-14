# Examples
load("//mace:mace.bzl", "if_android", "if_neon_enabled", "if_openmp_enabled")

cc_binary(
    name = "helloworld",
    srcs = [
        "helloworld.cc",
    ],
    linkopts = if_openmp_enabled(["-fopenmp"]),
    deps = [
        "//mace/core",
        "//mace/ops",
    ],
)

cc_test(
    name = "benchmark_example",
    testonly = 1,
    srcs = ["benchmark_example.cc"],
    linkopts = if_openmp_enabled(["-fopenmp"]),
    linkstatic = 1,
    deps = [
        "//mace/core",
        "//mace/core:test_benchmark_main",
    ],
)

cc_binary(
    name = "mace_run",
    srcs = ["mace_run.cc"],
    linkopts = if_openmp_enabled(["-fopenmp"]),
    linkstatic = 1,
    deps = [
        "//mace/codegen:generated_models",
        "//external:gflags_nothreads",
    ],
)
