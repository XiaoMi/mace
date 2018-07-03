# Examples
load("//mace:mace.bzl", "if_openmp_enabled", "if_android", "if_hexagon_enabled")

cc_binary(
    name = "example_static",
    srcs = ["example.cc"],
    copts = [
        "-Werror",
        "-Wextra",
    ] + if_android([
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
        "//mace:libmace_static",
    ] + if_hexagon_enabled([
        "//third_party/nnlib:libhexagon",
    ]),
)

cc_binary(
    name = "example_shared",
    srcs = ["example.cc"],
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + if_android([
        "-DMACE_ENABLE_OPENCL",
    ]),
    linkopts = ["-lm", "-pie", "-fPIE"] + if_openmp_enabled(["-fopenmp"]),
    linkstatic = 0,
    deps = [
        "//external:gflags_nothreads",
        "//mace/codegen:generated_mace_engine_factory",
        "//mace/utils:utils",
        "//mace:libmace",
    ],
)

