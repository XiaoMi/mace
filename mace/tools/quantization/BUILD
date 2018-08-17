# Quantize stat build

cc_binary(
    name = "quantize_stat",
    srcs = ["quantize_stat.cc"],
    copts = [
        "-Werror",
        "-Wextra",
    ],
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        "//external:gflags_nothreads",
        "//mace/codegen:generated_mace_engine_factory",
        "//mace/codegen:libmodels",
        "//mace/libmace",
    ],
)
