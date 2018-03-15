# Description:
# Mace core.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//mace:mace.bzl",
    "if_android",
    "if_hexagon_enabled",
    "if_not_hexagon_enabled",
    "if_production_mode",
    "if_not_production_mode",
    "if_openmp_enabled",
)

cc_library(
    name = "core",
    srcs = glob(
        [
            "*.cc",
            "runtime/cpu/*.cc",
            "runtime/opencl/*.cc",
            "runtime/hexagon/*.cc",
        ],
        exclude = [
            "runtime/opencl/opencl_production.cc",
            "runtime/opencl/opencl_development.cc",
            "*_test.cc",
            "runtime/hexagon/hexagon_controller_dummy.cc",
        ],
    ) + if_not_hexagon_enabled([
        "runtime/hexagon/hexagon_controller_dummy.cc",
    ]) + if_hexagon_enabled([
        "runtime/hexagon/libhexagon_controller.so",
    ]),
    hdrs = glob([
        "*.h",
        "runtime/cpu/*.h",
        "runtime/opencl/*.h",
        "runtime/hexagon/*.h",
    ]),
    copts = if_openmp_enabled(["-fopenmp"]),
    linkopts = ["-ldl"] + if_android([
        "-pie",
        "-lm",
    ]),
    deps = [
        ":opencl_headers",
        "//mace/utils",
        "//mace/codegen:generated_version",
    ] + if_production_mode([
        "//mace/utils:utils_prod",
        "//mace/core:opencl_prod",
    ]) + if_not_production_mode([
        "//mace/utils:utils_dev",
        "//mace/core:opencl_dev",
    ]),
)

cc_library(
    name = "opencl_headers",
    hdrs = glob([
        "runtime/opencl/*.h",
    ]),
    deps = [
        "@opencl_clhpp//:opencl_clhpp",
        "@opencl_headers//:opencl20_headers",
    ],
)

cc_library(
    name = "test_benchmark_main",
    testonly = 1,
    srcs = [
        "testing/test_benchmark.cc",
        "testing/test_benchmark_main.cc",
    ],
    hdrs = [
        "testing/test_benchmark.h",
    ],
    deps = [
        ":core",
        "//mace/utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "opencl_dev",
    srcs = ["runtime/opencl/opencl_development.cc"],
    linkstatic = 1,
    deps = [
        ":opencl_headers",
        "//mace/codegen:generated_opencl_dev",
        "//mace/utils",
    ],
)

cc_library(
    name = "opencl_prod",
    srcs = ["runtime/opencl/opencl_production.cc"],
    linkstatic = 1,
    deps = [
        ":opencl_headers",
        "//mace/codegen:generated_opencl_prod",
        "//mace/codegen:generated_tuning_params",
        "//mace/utils",
    ],
)
