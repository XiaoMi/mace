# Examples
load("//mace:mace.bzl", "if_android")

cc_binary(
    name = "helloworld",
    srcs = [
        "helloworld.cc",
        ],
    deps = [
        "//mace/core:core",
        ],
    linkopts = if_android(["-pie", "-llog"])
    )
