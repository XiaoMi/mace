# Description:
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//mace:mace.bzl",
    "if_android",
    "if_neon_enabled",
    "if_openmp_enabled",
    "if_android_armv7",
    "if_hexagon_enabled",
    "if_opencl_enabled",
    "if_quantize_enabled",
)

cc_library(
    name = "internal_ops",
    srcs = glob(
        [
            "*.cc",
            "arm/*.cc",
        ],
        exclude = [
            "*_test.cc",
            "*_benchmark.cc",
            "arm/*_test.cc",
            "ops_registry.cc",
            "ops_test_util.cc",
            "buffer_transform.cc",
            "lstm_cell.cc",
            "quantize.cc",
        ],
    ) + if_opencl_enabled(glob(
        [
            "opencl/*.cc",
            "opencl/image/*.cc",
            "opencl/buffer/*.cc",
            "buffer_transform.cc",
            "lstm_cell.cc",
        ],
        exclude = [
            "opencl/*_test.cc",
        ],
    )) + if_quantize_enabled(glob(
        [
            "quantize.cc",
        ],
    )),
    hdrs = glob(
        [
            "*.h",
            "arm/*.h",
        ],
        exclude = [
            "ops_registry.h",
            "ops_test_util.h",
            "fixpoint.h",
            "gemmlowp_util.h",
            "arm/fixpoint_*.h",
        ],
    ) + if_opencl_enabled(glob([
        "opencl/*.h",
        "opencl/image/*.h",
        "opencl/buffer/*.h",
    ])) + if_quantize_enabled(glob([
        "fixpoint.h",
        "gemmlowp_util.h",
        "arm/fixpoint_*.h",
    ])),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + if_openmp_enabled([
        "-fopenmp",
    ]) + if_neon_enabled([
        "-DMACE_ENABLE_NEON",
    ]) + if_android_armv7([
        "-mfpu=neon",
    ]) + if_android_armv7([
        "-mfloat-abi=softfp",
    ]) + if_opencl_enabled([
        "-DMACE_ENABLE_OPENCL",
    ]) + if_quantize_enabled([
        "-DMACE_ENABLE_QUANTIZE",
    ]) + if_hexagon_enabled([
        "-DMACE_ENABLE_HEXAGON",
    ]),
    linkopts = if_android(["-lm"]),
    deps = [
        "//mace/core",
        "@tflite",
    ] + if_quantize_enabled([
        "@gemmlowp",
    ]),
)

cc_library(
    name = "ops",
    srcs = [
        "ops_registry.cc",
    ],
    hdrs = [
        "ops_registry.h",
    ],
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + if_openmp_enabled([
        "-fopenmp",
    ]) + if_neon_enabled([
        "-DMACE_ENABLE_NEON",
    ]) + if_android_armv7([
        "-mfpu=neon",
    ]) + if_android_armv7([
        "-mfloat-abi=softfp",
    ]) + if_opencl_enabled([
        "-DMACE_ENABLE_OPENCL",
    ]) + if_quantize_enabled([
        "-DMACE_ENABLE_QUANTIZE",
    ]) + if_hexagon_enabled([
        "-DMACE_ENABLE_HEXAGON",
    ]),
    linkopts = if_android(["-lm"]),
    deps = [
        "internal_ops",
    ],
)

cc_library(
    name = "test",
    testonly = 1,
    srcs = [
        "ops_test_util.cc",
    ],
    hdrs = glob([
        "*_test_util.h",
    ]),
    copts = [
        "-Werror",
        "-Wextra",
    ] + if_openmp_enabled(["-fopenmp"]) + if_neon_enabled([
        "-DMACE_ENABLE_NEON",
    ]) + if_android_armv7([
        "-mfpu=neon",
    ]) + if_android_armv7([
        "-mfloat-abi=softfp",
    ]) + if_opencl_enabled([
        "-DMACE_ENABLE_OPENCL",
    ]) + if_hexagon_enabled([
        "-DMACE_ENABLE_HEXAGON",
    ]),
    deps = [
        "ops",
        "@gtest",
    ],
)

cc_test(
    name = "ops_test",
    testonly = 1,
    srcs = glob(
        [
            "*_test.cc",
            "arm/*_test.cc",
            "opencl/*_test.cc",
        ],
        exclude = [
            "fixpoint_test.cc",
        ],
    ) + if_quantize_enabled(glob(
        [
            "fixpoint_test.cc",
        ],
    )),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + if_openmp_enabled([
        "-fopenmp",
    ]) + if_neon_enabled([
        "-DMACE_ENABLE_NEON",
    ]) + if_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + if_opencl_enabled([
        "-DMACE_ENABLE_OPENCL",
    ]) + if_quantize_enabled([
        "-DMACE_ENABLE_QUANTIZE",
    ]) + if_hexagon_enabled([
        "-DMACE_ENABLE_HEXAGON",
    ]),
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        "test",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "ops_benchmark",
    testonly = 1,
    srcs = glob(["*_benchmark.cc"]),
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + if_openmp_enabled([
        "-fopenmp",
    ]) + if_neon_enabled([
        "-DMACE_ENABLE_NEON",
    ]) + if_android_armv7([
        "-mfpu=neon",
        "-mfloat-abi=softfp",
    ]) + if_opencl_enabled([
        "-DMACE_ENABLE_OPENCL",
    ]) + if_quantize_enabled([
        "-DMACE_ENABLE_QUANTIZE",
    ]) + if_hexagon_enabled([
        "-DMACE_ENABLE_HEXAGON",
    ]),
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        "test",
        "//mace/core:test_benchmark_main",
        "//third_party/eigen3",
    ],
)
