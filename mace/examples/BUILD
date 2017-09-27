# Examples
load("//mace:mace.bzl", "if_android")

cc_binary(
    name = "helloworld",
    srcs = [
        "helloworld.cc",
    ],
    deps = [
        "//mace/core",
        "//mace/ops",
        "@org_tensorflow//tensorflow/core:android_tensorflow_lib",
    ],
    copts = ["-std=c++11"],
    linkopts = ["-fopenmp",] + if_android(["-ldl"]),
)

cc_test(
    name = "benchmark_example",
    srcs = ["benchmark_example.cc"],
    deps = [
        "//mace/core",
        "//mace/core:test_benchmark_main",
    ],
    copts = ["-std=c++11"],
    linkopts = ["-fopenmp",] + if_android(["-ldl"]),
    linkstatic = 1,
    testonly = 1,
)

cc_binary(
    name = "mace_run",
    srcs = [
        "mace_run.cc",
    ],
    deps = [
        "//mace/core",
        "//mace/utils",
        "//mace/ops",
    ],
    copts = ["-std=c++11",],
    linkopts = ["-fopenmp",] + if_android(["-ldl"]),
    linkstatic = 1,
)
