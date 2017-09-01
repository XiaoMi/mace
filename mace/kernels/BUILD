# Description:
# Mace neon kernels.
#
package(
    default_visibility = ["//visibility:public"],
)


licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android")

cc_library(
    name = "kernels",
    srcs = glob(["*.cc"]) + if_android(glob(["neon/*.cc"])),
    hdrs = glob(["*.h"]) + if_android(glob(["neon/*.h"])),
    deps = [
        "//mace/core:core",
    ],
    copts = ['-std=c++11'],
)

