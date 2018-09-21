package(default_visibility = ["//tools/aarch64_compiler:__pkg__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "gcc",
    srcs = [
        "aarch64-linux-gnu-gcc",
        "@gcc_linaro_7_3_1_aarch64_linux_gnu//:gcc",
    ],
)

filegroup(
    name = "ar",
    srcs = [
        "aarch64-linux-gnu-ar",
        "@gcc_linaro_7_3_1_aarch64_linux_gnu//:ar",
    ],
)

filegroup(
    name = "ld",
    srcs = [
        "aarch64-linux-gnu-ld",
        "@gcc_linaro_7_3_1_aarch64_linux_gnu//:ld",
    ],
)

filegroup(
    name = "nm",
    srcs = [
        "aarch64-linux-gnu-nm",
        "@gcc_linaro_7_3_1_aarch64_linux_gnu//:nm",
    ],
)

filegroup(
    name = "objcopy",
    srcs = [
        "aarch64-linux-gnu-objcopy",
        "@gcc_linaro_7_3_1_aarch64_linux_gnu//:objcopy",
    ],
)

filegroup(
    name = "objdump",
    srcs = [
        "aarch64-linux-gnu-objdump",
        "@gcc_linaro_7_3_1_aarch64_linux_gnu//:objdump",
    ],
)

filegroup(
    name = "strip",
    srcs = [
        "aarch64-linux-gnu-strip",
        "@gcc_linaro_7_3_1_aarch64_linux_gnu//:strip",
    ],
)

filegroup(
    name = "as",
    srcs = [
        "aarch64-linux-gnu-as",
        "@gcc_linaro_7_3_1_aarch64_linux_gnu//:as",
    ],
)

filegroup(
    name = "tool-wrappers",
    srcs = [
        ":ar",
        ":as",
        ":gcc",
        ":ld",
        ":nm",
        ":objcopy",
        ":objdump",
        ":strip",
    ],
)
