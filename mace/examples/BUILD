# Examples
load("//mace:mace.bzl", "if_android", "if_enable_neon", "if_embed_binary_program")

cc_binary(
    name = "helloworld",
    srcs = [
        "helloworld.cc",
    ],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1"],
    linkopts = if_enable_neon(["-fopenmp"]),
    deps = [
        "//mace/core",
        "//mace/ops",
        "//mace/core:opencl_runtime",
    ],
)

cc_test(
    name = "benchmark_example",
    testonly = 1,
    srcs = ["benchmark_example.cc"],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1"],
    linkopts = if_enable_neon(["-fopenmp"]),
    linkstatic = 1,
    deps = [
        "//mace/core",
        "//mace/core:test_benchmark_main",
    ],
)

cc_binary(
    name = "mace_run",
    srcs = ["mace_run.cc"],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1"],
    linkopts = if_enable_neon(["-fopenmp"]),
    linkstatic = 1,
    deps = [
        "//mace/codegen:generated_models_lib",
        "//mace/utils:command_line_flags",
    ],
)
