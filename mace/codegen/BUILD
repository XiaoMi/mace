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
    ] + if_embed_binary_program(["//mace/core:opencl_binary_linkage"]) +
    if_use_source_program(["//mace/core:opencl_source_linkage"]) + [
        "//mace/codegen:version_lib",
        "//mace/codegen:generated_opencl_source_lib",
        "//mace/codegen:generated_opencl_compiled_lib",
        "//mace/codegen:generated_tuning_lib",
    ],
)

cc_library(
    name = "generated_opencl_source_lib",
    srcs = glob(["opencl/opencl_encrypt_program.cc"]),
    copts = ["-std=c++11", "-D_GLIBCXX_USE_C99_MATH_TR1", "-Werror=return-type"],
    linkstatic = 1,
)

cc_library(
    name = "generated_opencl_compiled_lib",
    srcs = glob(["opencl/opencl_compiled_program.cc"]),
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
