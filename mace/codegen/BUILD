# Description:
# Generated model and runtime code.
#
package(
    default_visibility = ["//visibility:public"],
)

load("//mace:mace.bzl", "if_embed_binary_program")

cc_library(
    name = "generated_models_lib",
    srcs = glob(["models/*/*.cc"]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1"],
    linkstatic = 1,
    deps = [
        "//mace/core",
        "//mace/ops",
    ] + if_embed_binary_program([
        '//mace/codegen:generated_opencl_lib',
        '//mace/codegen:generated_tuning_lib',
    ]),
)

cc_library(
    name = "generated_opencl_lib",
    srcs = glob(["opencl/*.cc"]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1"],
    linkstatic = 1,
)

cc_library(
    name = "generated_tuning_lib",
    srcs = glob(["tuning/*.cc"]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1"],
    linkstatic = 1,
)
