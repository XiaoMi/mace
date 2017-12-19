# Examples
load("//mace:mace.bzl", "if_android")

cc_binary(
    name = "helloworld",
    srcs = [
        "helloworld.cc",
    ],
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1"],
    linkopts = ["-fopenmp"],
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
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        "//mace/core",
        "//mace/core:test_benchmark_main",
    ],
)

cc_binary(
    name = "mace_run",
    srcs = glob(["models/*/*.cc"] + ["mace_run.cc"]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1"],
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        "//mace/core",
        "//mace/ops",
        "//mace/utils:command_line_flags",
    ],
)
