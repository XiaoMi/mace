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

config_setting(
    name = "opencl_enabled",
    define_values = {
        "opencl": "true",
    },
    visibility = ["//visibility:public"],
)
