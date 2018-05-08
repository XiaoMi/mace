# Examples
load("//mace:mace.bzl", "if_openmp_enabled", "if_android")

cc_binary(
    name = "mace_run",
    srcs = ["mace_run.cc"],
    linkopts = if_openmp_enabled(["-fopenmp"]),
    linkstatic = 1,
    copts = if_android(["-DMACE_ENABLE_OPENCL"]),
    deps = [
        "//external:gflags_nothreads",
        "//mace/codegen:generated_models",
        "//mace/codegen:generated_mace_engine_creator",
        "//mace/core:core",
    ],
)
