# Description:
# Mace operators.
#
package(
    default_visibility = ["//visibility:public"],
)


licenses(["notice"])  # Apache 2.0

cc_library(
    name = "op",
    srcs = ["relu.cc"],
    hdrs = glob(["*.h"]),
    deps = [
        "//mace/proto:cc_proto",
        "//mace/core:core",
    ],
)

cc_test(
    name = "op_test",
    srcs = ["relu_test.cc",],
    deps = [
        "@gtest//:gtest",
        ":op",
    ],
)