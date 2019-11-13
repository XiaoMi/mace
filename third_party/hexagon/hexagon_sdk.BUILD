package(default_visibility = ['//visibility:public'])

filegroup(
    name = 'sdk_location',
    srcs = ["readme.txt"],
)

cc_library(
    name = 'headers_incs',
    hdrs = glob([
        "incs/*.h",
    ]),
    strip_include_prefix = "incs/",
)

cc_library(
    name = 'headers_incs_stddef',
    hdrs = glob([
        "incs/stddef/*.h",
    ]),
    strip_include_prefix = "incs/stddef/",
)

cc_library(
    name = 'headers_dsp',
    hdrs = glob([
        "libs/common/remote/ship/hexagon_Release_toolv81_v60/*.h",
    ]),
    strip_include_prefix = "libs/common/remote/ship/hexagon_Release_toolv81_v60/",
    deps = [
        ":headers_incs",
        ":headers_incs_stddef",
        "@hexagon_tools//:headers_tools_target",
    ],
)

cc_library(
    name = 'headers_arm',
    hdrs = glob([
        "libs/common/remote/ship/android_Release_aarch64/*.h",
    ]),
    strip_include_prefix = "libs/common/remote/ship/android_Release_aarch64/",
    deps = [
        ":headers_incs",
        ":headers_incs_stddef",
    ],
)

cc_library(
    name = 'sdk_arm',
    srcs = glob([
        "libs/common/remote/ship/android_Release_aarch64/libcdsprpc.so",
        "libs/common/rpcmem/rpcmem.a",
    ]),
    deps = [
        ":headers_arm",
    ],
)