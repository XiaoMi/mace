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
        ],
        exclude = [
            "*_test.cc",
        ],
    ) + if_android(glob(
        [
            "runtime/opencl/*.cc",
        ],
    )) + if_hexagon_enabled(glob([
        "runtime/hexagon/*.cc",
    ])),
    hdrs = glob([
        "*.h",
        "runtime/cpu/*.h",
    ]) + if_android(glob(
        [
            "runtime/opencl/*.h",
        ],
    )) + if_hexagon_enabled(glob(["runtime/hexagon/*.h"])),
    copts = if_openmp_enabled(["-fopenmp"]) + if_android([
        "-DMACE_ENABLE_OPENCL",
    ]) + if_hexagon_enabled(["-DMACE_ENABLE_HEXAGON"]),
    linkopts = ["-ldl"] + if_android([
        "-pie",
        "-lm",
    ]),
    deps = [
        "//mace/utils",
        "//mace/codegen:generated_version",
    ] + if_android([
        ":opencl_headers",
        "//mace/codegen:generated_opencl",
        "@half//:half",
    ]) + if_production_mode([
        "//mace/codegen:generated_tuning_params",
        "//mace/utils:utils_prod",
    ]) + if_not_production_mode([
        "//mace/utils:utils_dev",
    ]) + if_hexagon_enabled([
        "//mace/third_party/nnlib:libhexagon",
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
        "//external:gflags_nothreads",
        "//mace/utils",
    ],
    alwayslink = 1,
)
