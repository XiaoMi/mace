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
    "if_openmp_enabled",
    "if_neon_enabled",
    "if_opencl_enabled",
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
    ) + if_opencl_enabled(glob(
        [
            "runtime/opencl/*.cc",
        ],
    )) + if_hexagon_enabled(glob([
        "runtime/hexagon/*.cc",
    ])),
    hdrs = glob([
        "*.h",
        "runtime/cpu/*.h",
    ]) + if_opencl_enabled(glob(
        [
            "runtime/opencl/*.h",
        ],
    )) + if_hexagon_enabled(glob(["runtime/hexagon/*.h"])),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + if_openmp_enabled([
        "-fopenmp",
        "-DMACE_ENABLE_OPENMP",
    ]) + if_opencl_enabled([
        "-DMACE_ENABLE_OPENCL",
    ]) + if_hexagon_enabled([
        "-DMACE_ENABLE_HEXAGON",
    ]) + if_neon_enabled([
        "-DMACE_ENABLE_NEON",
    ]),
    linkopts = ["-ldl"] + if_android([
        "-pie",
        "-lm",
    ]),
    deps = [
        "//mace/codegen:generated_version",
        "//mace/proto:mace_cc",
        "//mace/utils",
    ] + if_opencl_enabled([
        ":opencl_headers",
        "//mace/codegen:generated_opencl",
        "@half//:half",
    ]) + if_hexagon_enabled([
        "//third_party/nnlib:libhexagon",
    ]),
)

cc_library(
    name = "opencl_headers",
    hdrs = glob([
        "runtime/opencl/*.h",
    ]),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ],
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
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ],
    deps = [
        ":core",
        "//external:gflags_nothreads",
        "//mace/utils",
    ],
)
