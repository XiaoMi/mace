# Examples
load(
    "//:mace.bzl",
    "if_neon_enabled",
    "if_production_mode",
    "if_not_production_mode",
    "if_hexagon_enabled",
    "if_openmp_enabled",
)

cc_binary(
    name = "mace_run",
    srcs = ["mace_run.cc"],
    linkopts = if_openmp_enabled(["-fopenmp"]),
    linkstatic = 1,
    deps = [
        "//codegen:generated_models",
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
