# Examples
load("//mace:mace.bzl", "if_android")

cc_binary(
    name = "helloworld",
    srcs = [
        "helloworld.cc",
        ],
    deps = [
        "//mace/ops:ops",
        ],
    copts = ['-std=c++11'],
    linkopts = if_android(["-pie", "-llog"]),
)
