# Examples
load("//mace:mace.bzl", "if_android")

cc_binary(
    name = "helloworld",
    srcs = [
        "helloworld.cc",
    ],
    copts = ["-std=c++11"],
    linkopts = ["-fopenmp"] + if_android(["-ldl"]),
    deps = [
        "//mace/core",
        "//mace/ops",
        "@org_tensorflow//tensorflow/core:android_tensorflow_lib",
    ],
)

cc_test(
    name = "benchmark_example",
    testonly = 1,
    srcs = ["benchmark_example.cc"],
    copts = ["-std=c++11"],
    linkopts = ["-fopenmp"] + if_android(["-ldl"]),
    linkstatic = 1,
    deps = [
        "//mace/core",
        "//mace/core:test_benchmark_main",
    ],
)

cc_binary(
    name = "mace_run",
    srcs = [
        "mace_run.cc",
    ],
    copts = ["-std=c++11"],
    linkopts = ["-fopenmp"] + if_android(["-ldl"]),
    linkstatic = 1,
    deps = [
        "//mace/core",
        "//mace/ops",
        "//mace/utils:command_line_flags",
    ],
)
