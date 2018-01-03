# Description:
# Generated model and runtime code.
#
package(
    default_visibility = ["//visibility:public"],
)

load("//mace:mace.bzl", "if_embed_binary_program", "if_use_source_program")

cc_library(
    name = "generated_models_lib",
    srcs = glob(["models/*/*.cc"]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    linkstatic = 1,
    deps = [
        "//mace/core",
        "//mace/ops",
        "//mace/codegen:distribute_lib",
        "//mace/codegen:version_lib",
        "//mace/codegen:generated_opencl_lib",
        "//mace/codegen:generated_tuning_lib",
    ],
)

cc_library(
    name = "distribute_lib",
    srcs = if_embed_binary_program(["distribute/use_binary.cc"]) +
        if_use_source_program(["distribute/use_source.cc"]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    linkstatic = 1,
    deps = [
        "//mace/core",
    ],
)

cc_library(
    name = "generated_opencl_lib",
    srcs = glob(["opencl/*.cc"]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    linkstatic = 1,
)

cc_library(
    name = "generated_tuning_lib",
    srcs = glob(["tuning/*.cc"]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    linkstatic = 1,
)

cc_library(
    name = "version_lib",
    srcs = glob(["version/*.cc"]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    linkstatic = 1,
    deps = [
        "//mace/core",
    ]
)
