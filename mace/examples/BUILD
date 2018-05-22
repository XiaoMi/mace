# Examples
load("//mace:mace.bzl", "if_openmp_enabled")

cc_binary(
    name = "example",
    srcs = ["example.cc"],
    linkopts = if_openmp_enabled(["-fopenmp"]),
    linkstatic = 1,
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
    deps = [
        "//external:gflags_nothreads",
        "//mace/codegen:generated_models",
        "//mace/codegen:generated_mace_engine_factory",
    ],
)
