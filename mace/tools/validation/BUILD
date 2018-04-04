# Examples
load("//mace:mace.bzl", "if_openmp_enabled")

cc_binary(
    name = "mace_run",
    srcs = ["mace_run.cc"],
    linkopts = if_openmp_enabled(["-fopenmp"]),
    linkstatic = 1,
    deps = [
        "//external:gflags_nothreads",
        "//mace/codegen:generated_models",
        "//mace/core:core",
    ],
)
