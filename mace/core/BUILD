# Description:
# Mace core.
#
package(
    default_visibility = ["//visibility:public"],
)


licenses(["notice"])  # Apache 2.0

cc_library(
    name = "core",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//mace/proto:cc_proto",
    ],
    copts = ['-std=c++11'],
)

