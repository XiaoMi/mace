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
    "if_quantize_enabled",
)

cc_library(
    name = "mace_api_test_header",
    hdrs = [
        "mace_api_test.h",
    ],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
)

cc_test(
    name = "mace_api_test",
    testonly = 1,
    srcs = ["mace_api_test.cc"],
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + if_openmp_enabled([
        "-fopenmp",
        "-DMACE_ENABLE_OPENMP",
    ]) + if_opencl_enabled([
        "-DMACE_ENABLE_OPENCL",
    ]) + if_quantize_enabled([
        "-DMACE_ENABLE_QUANTIZE",
    ]) + if_hexagon_enabled([
        "-DMACE_ENABLE_HEXAGON",
    ]) + if_neon_enabled([
        "-DMACE_ENABLE_NEON",
    ]),
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        ":mace_api_test_header",
        "//mace/ops:test",
        "//mace/libmace:libmace",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "mace_api_mt_test",
    testonly = 1,
    srcs = ["mace_api_mt_test.cc"],
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + if_openmp_enabled([
        "-fopenmp",
        "-DMACE_ENABLE_OPENMP",
    ]) + if_opencl_enabled([
        "-DMACE_ENABLE_OPENCL",
    ]) + if_quantize_enabled([
        "-DMACE_ENABLE_QUANTIZE",
    ]) + if_hexagon_enabled([
        "-DMACE_ENABLE_HEXAGON",
    ]) + if_neon_enabled([
        "-DMACE_ENABLE_NEON",
    ]),
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        ":mace_api_test_header",
        "//mace/ops:test",
        "//mace/libmace:libmace",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "mace_api_exception_test",
    testonly = 1,
    srcs = ["mace_api_exception_test.cc"],
    copts = [
        "-Werror",
        "-Wextra",
        "-Wno-missing-field-initializers",
    ] + if_openmp_enabled([
        "-fopenmp",
        "-DMACE_ENABLE_OPENMP",
    ]) + if_opencl_enabled([
        "-DMACE_ENABLE_OPENCL",
    ]) + if_quantize_enabled([
        "-DMACE_ENABLE_QUANTIZE",
    ]) + if_hexagon_enabled([
        "-DMACE_ENABLE_HEXAGON",
    ]) + if_neon_enabled([
        "-DMACE_ENABLE_NEON",
    ]),
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        "//mace/ops:test",
        "//mace/libmace:libmace",
        "@gtest//:gtest_main",
    ],
)
