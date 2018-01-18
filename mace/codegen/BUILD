# Description:
# Generated model and runtime code.
#
package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "generated_models",
    srcs = glob(["models/*/*.cc"]),
    linkstatic = 1,
    deps = [
        "//mace/core",
        "//mace/ops",
    ],
)

cc_library(
    name = "generated_opencl_dev",
    srcs = ["opencl/opencl_encrypt_program.cc"],
    linkstatic = 1,
)

cc_library(
    name = "generated_opencl_prod",
    srcs = ["opencl/opencl_compiled_program.cc"],
    linkstatic = 1,
)

cc_library(
    name = "generated_tuning_params",
    srcs = ["tuning/tuning_params.cc"],
    linkstatic = 1,
)


cc_library(
    name = "generated_version",
    srcs = ["version/version.cc"],
    linkstatic = 1,
)
