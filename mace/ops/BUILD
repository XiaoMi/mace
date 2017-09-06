# Description:
# Mace operators.
#
package(
    default_visibility = ["//visibility:public"],
)


licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android")

cc_library(
    name = "ops",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//mace/proto:cc_proto",
        "//mace/core:core",
        "//mace/kernels:kernels",
    ],
    copts = ['-std=c++11'],
    alwayslink = 1,
)
