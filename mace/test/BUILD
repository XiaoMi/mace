# Description:
# Mace operators.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_test(
    name = "mace_api_test",
    testonly = 1,
    srcs = glob(
        ["mace_api_test.cc"],
    ),
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        "//mace/core:core",
        "//mace/kernels:kernels",
        "//mace/ops:ops",
        "//mace/ops:test",
        "@gtest//:gtest_main",
    ],
)
