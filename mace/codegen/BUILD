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
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
    deps = [
        "//mace/core",
        "//mace/ops",
    ],
)

cc_library(
    name = "generated_opencl",
    srcs = glob(["opencl/*.cc"]),
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
)

cc_library(
    name = "generated_tuning_params",
    srcs = ["tuning/tuning_params.cc"],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
)

cc_library(
    name = "generated_version",
    srcs = ["version/version.cc"],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
)

cc_library(
    name = "generated_mace_engine_factory",
    hdrs = ["engine/mace_engine_factory.h"],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
    deps = [
        "//mace/public",
    ],
)
