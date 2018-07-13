# Description:
# Mace operators.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android", "if_neon_enabled", "if_openmp_enabled", "if_android_armv7", "if_hexagon_enabled")

cc_test(
    name = "mace_api_test",
    testonly = 1,
    srcs = ["mace_api_test.cc"],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"] +
      if_openmp_enabled(["-fopenmp"]) +
      if_neon_enabled(["-DMACE_ENABLE_NEON"]) +
      if_android_armv7(["-mfpu=neon"]) +
      if_android_armv7(["-mfloat-abi=softfp"]) +
      if_android(["-DMACE_ENABLE_OPENCL"]) +
      if_hexagon_enabled(["-DMACE_ENABLE_HEXAGON"]),
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        "//mace/ops:test",
        "//mace/libmace:libmace",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "mace_api_mt_test",
    testonly = 1,
    srcs = ["mace_api_mt_test.cc"],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"] +
      if_openmp_enabled(["-fopenmp"]) +
      if_neon_enabled(["-DMACE_ENABLE_NEON"]) +
      if_android_armv7(["-mfpu=neon"]) +
      if_android_armv7(["-mfloat-abi=softfp"]) +
      if_android(["-DMACE_ENABLE_OPENCL"]) +
      if_hexagon_enabled(["-DMACE_ENABLE_HEXAGON"]),
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        "//mace/ops:test",
        "//mace/libmace:libmace",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "mace_api_exception_test",
    testonly = 1,
    srcs = ["mace_api_exception_test.cc"],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"] +
      if_openmp_enabled(["-fopenmp"]) +
      if_neon_enabled(["-DMACE_ENABLE_NEON"]) +
      if_android_armv7(["-mfpu=neon"]) +
      if_android_armv7(["-mfloat-abi=softfp"]) +
      if_android(["-DMACE_ENABLE_OPENCL"]) +
      if_hexagon_enabled(["-DMACE_ENABLE_HEXAGON"]),
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        "//mace/ops:test",
        "//mace/libmace:libmace",
        "@gtest//:gtest_main",
    ],
)
