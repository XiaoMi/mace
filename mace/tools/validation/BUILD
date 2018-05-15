# Examples
load("//mace:mace.bzl", "if_openmp_enabled", "if_android")

cc_binary(
    name = "mace_run",
    srcs = ["mace_run.cc"],
    copts = if_android([
        "-DMACE_ENABLE_OPENCL",
    ]),
    linkopts = if_openmp_enabled(["-fopenmp"]),
    linkstatic = 1,
    deps = [
        "//external:gflags_nothreads",
        "//mace/codegen:generated_mace_engine_factory",
        "//mace/codegen:generated_models",
        "//mace/core",
    ],
)
