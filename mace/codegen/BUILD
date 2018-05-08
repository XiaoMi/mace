# Description:
# Generated model and runtime code.
#
package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "generated_models",
    srcs = glob(["models/*/*.cc"]),
    hdrs = glob(["models/*/*.h"]),
    linkstatic = 1,
    deps = [
        "//mace/core",
        "//mace/ops",
    ],
)

cc_library(
    name = "generated_opencl",
    srcs = glob(["opencl/*.cc"]),
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

cc_library(
    name = "generated_mace_engine_creator",
    srcs = ["engine/mace_engine_creator.cc"],
    linkstatic = 1,
    deps = [
        ":generated_models",
        "//mace/public",
        "//mace/utils",
    ],
)
