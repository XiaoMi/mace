config_setting(
    name = "android",
    values = {
        "crosstool_top": "//external:android/crosstool",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_armv7",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "armeabi-v7a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "arm64-v8a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "neon_enabled",
    define_values = {
        "neon": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "hexagon_enabled",
    define_values = {
        "hexagon": "true",
    },
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "armeabi-v7a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "openmp_enabled",
    define_values = {
        "openmp": "true",
    },
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libmace.so",
    linkopts = [
        "-Wl,-soname,libmace.so",
        "-Wl,--version-script",
        "mace_version_script.lds",
    ],
    linkshared = 1,
    linkstatic = 0,
    deps = [
        ":mace_version_script.lds",
        "//mace/ops",
    ],
)

cc_library(
    name = "libmace",
    srcs = ["libmace.so"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "libmace_static",
    srcs = [
        "//mace/codegen:generated_opencl",
        "//mace/codegen:generated_version",
        "//mace/core",
        "//mace/kernels",
        "//mace/ops",
        "//mace/utils",
        "//mace/proto:mace_cc",
        "@com_google_protobuf//:protobuf_lite",
    ],
    outs = ["libmace.a"],
    cmd = "tmp_mri_file=$$(mktemp mace-static-lib-mri.XXXXXXXXXX);" +
          "mri_stream=$$(python $(location //mace/python/tools:archive_static_lib) " +
          "$(locations //mace/codegen:generated_opencl) " +
          "$(locations //mace/codegen:generated_version) " +
          "$(locations //mace/core:core) " +
          "$(locations //mace/kernels:kernels) " +
          "$(locations //mace/ops:ops) " +
          "$(locations //mace/utils:utils) " +
          "$(locations //mace/proto:mace_cc) " +
          "$(locations @com_google_protobuf//:protobuf_lite) " +
          "$@ " +
          "$$tmp_mri_file);" +
          "$(AR) -M <$$tmp_mri_file;" +
          "rm -rf $$tmp_mri_file;",
    tools = ["//mace/python/tools:archive_static_lib"],
    visibility = ["//visibility:public"],
)
