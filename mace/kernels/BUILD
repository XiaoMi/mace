# Description:
# Mace neon kernels.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("//mace:mace.bzl", "if_android", "if_neon_enabled", "if_openmp_enabled", "if_android_armv7", "if_hexagon_enabled")

cc_library(
    name = "kernels",
    srcs = glob(
        [
            "*.cc",
            "arm/*.cc",
        ],
        exclude = [
            "*_test.cc",
            "arm/*_test.cc",
        ],
    ) + if_android(glob([
            "opencl/*.cc",
        ],
        exclude = [
            "opencl/*_test.cc",
        ])),
    hdrs = glob(
        [
            "*.h",
            "arm/*.h",
        ],
        exclude = [
            "buffer_to_image.h",
        ],
    ) + if_android(glob([
            "opencl/*.h",
            "buffer_to_image.h",
        ])),
    copts = ["-Werror"] +
      if_openmp_enabled(["-fopenmp"]) +
      if_neon_enabled(["-DMACE_ENABLE_NEON"]) +
      if_android_armv7(["-mfpu=neon"]) +
      if_android_armv7(["-mfloat-abi=softfp"]) +
      if_android(["-DMACE_ENABLE_OPENCL"]) +
      if_hexagon_enabled(["-DMACE_ENABLE_HEXAGON"]),
    linkopts = if_android(["-lm"]),
    deps = [
        "//mace/core",
        "//mace/utils",
    ],
)

cc_test(
    name = "kernels_test",
    testonly = 1,
    srcs = glob(
        [
            "*_test.cc",
            "arm/*_test.cc",
            "opencl/*_test.cc",
        ],
    ),
    copts = ["-Werror"] +
      if_openmp_enabled(["-fopenmp"]) +
      if_neon_enabled(["-DMACE_ENABLE_NEON"]) +
      if_android_armv7(["-mfpu=neon"]) +
      if_android_armv7(["-mfloat-abi=softfp"]) +
      if_android(["-DMACE_ENABLE_OPENCL"]) +
      if_hexagon_enabled(["-DMACE_ENABLE_HEXAGON"]),
    linkopts = ["-fopenmp"],
    linkstatic = 1,
    deps = [
        ":kernels",
        "@gtest//:gtest",
        "@gtest//:gtest_main",
    ],
)
