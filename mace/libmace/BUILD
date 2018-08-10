# Description:
# Mace libmace.
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
    "if_opencl_enabled_str",
)

cc_library(
    name = "libmace",
    srcs = glob(
        ["*.cc"],
    ),
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
        "//mace/public",
        "//mace/ops",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "libmace.so",
    linkopts = [
        "-Wl,-soname,libmace.so",
        "-Wl,--version-script",
        "mace_version_script.lds",
    ] + if_openmp_enabled([
        "-fopenmp",
    ]),
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":mace_version_script.lds",
        "//mace/libmace",
    ],
)

cc_library(
    name = "libmace_dynamic",
    srcs = ["libmace.so"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "libmace_static",
    srcs = [
        "//mace/codegen:generated_version",
        "//mace/core",
        "//mace/kernels",
        "//mace/ops",
        "//mace/libmace",
        "//mace/utils",
        "//mace/proto:mace_cc",
        "@com_google_protobuf//:protobuf_lite",
    ] + if_opencl_enabled([
        "//mace/codegen:generated_opencl",
    ]),
    outs = ["libmace.a"],
    cmd = "tmp_mri_file=$$(mktemp mace-static-lib-mri.XXXXXXXXXX);" +
          "mri_stream=$$(python $(location //mace/python/tools:archive_static_lib) " +
          "$(locations //mace/codegen:generated_version) " +
          "$(locations //mace/core:core) " +
          "$(locations //mace/kernels:kernels) " +
          "$(locations //mace/ops:ops) " +
          "$(locations //mace/libmace:libmace) " +
          "$(locations //mace/utils:utils) " +
          "$(locations //mace/proto:mace_cc) " +
          "$(locations @com_google_protobuf//:protobuf_lite) " +
          if_opencl_enabled_str("$(locations //mace/codegen:generated_opencl) ") +
          "$@ " +
          "$$tmp_mri_file);" +
          "$(AR) -M <$$tmp_mri_file;" +
          "rm -rf $$tmp_mri_file;" +
          "$(STRIP) -x $@;",
    tools = ["//mace/python/tools:archive_static_lib"],
    visibility = ["//visibility:public"],
)
