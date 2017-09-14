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
    ],
    copts = ["-std=c++11"],
    linkopts = if_android(["-ldl"]),
)

cc_test(
    name = "benchmark_example",
    srcs = ["benchmark_example.cc"],
    deps = [
        "//mace/core",
        "//mace/core:test_benchmark_main",
    ],
    copts = ["-std=c++11"],
    linkopts = if_android(["-ldl"]),
    linkstatic = 1,
    testonly = 1,
)
