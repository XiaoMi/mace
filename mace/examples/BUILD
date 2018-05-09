# Examples
load("//mace:mace.bzl", "if_openmp_enabled")

cc_binary(
    name = "example",
    srcs = ["example.cc"],
    linkopts = if_openmp_enabled(["-fopenmp"]),
    linkstatic = 1,
    deps = [
        "//external:gflags_nothreads",
        "//mace/codegen:generated_models",
        "//mace/codegen:generated_mace_engine_factory",
    ],
)
