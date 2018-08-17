# Examples
load("//mace:mace.bzl", "if_openmp_enabled", "if_android", "if_opencl_enabled")

cc_binary(
    name = "mace_run_static",
    srcs = ["mace_run.cc"],
    copts = [
        "-Werror",
        "-Wextra",
    ] + if_opencl_enabled([
        "-DMACE_ENABLE_OPENCL",
    ]),
    linkopts = if_openmp_enabled(["-fopenmp"]),
    linkstatic = 1,
    deps = [
        "//external:gflags_nothreads",
        "//mace/codegen:generated_mace_engine_factory",
        "//mace/codegen:libmodels",
        "//mace/libmace",
    ],
)

cc_binary(
    name = "mace_run_dynamic",
    srcs = ["mace_run.cc"],
    copts = [
        "-Werror",
        "-Wextra",
    ] + if_opencl_enabled([
        "-DMACE_ENABLE_OPENCL",
    ]),
    linkopts = [
        "-lm",
    ] + if_openmp_enabled([
        "-fopenmp"
    ]) + if_android([
        "-ldl",
        "-pie",
        "-llog",
    ]),
    linkstatic = 1,
    deps = [
        "//external:gflags_nothreads",
        "//mace/codegen:generated_mace_engine_factory",
        "//mace/codegen:libmodels",
        "//mace/libmace:libmace_dynamic",
    ],
)
