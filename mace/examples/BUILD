# Examples
load("//mace:mace.bzl", "if_android")

cc_binary(
    name = "helloworld",
    srcs = [
        "helloworld.cc",
    ],
    copts = ["-std=c++11"],
    linkopts = if_android([
        "-pie",
        "-llog",
    ]),
    deps = [
        "//mace/core",
        "//mace/ops",
    ],
)

cc_test(
    name = "benchmark_example",
    srcs = ["benchmark_example.cc"],
    copts = ["-std=c++11"],
    linkopts = if_android([
        "-pie",
        "-llog",
    ]),
    linkstatic = 1,
    deps = [
        "//mace/core",
        "//mace/core:test_main",
    ],
)
