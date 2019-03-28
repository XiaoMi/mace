package(default_visibility = ["//tools/arm_compiler:__pkg__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "gcc",
    srcs = [
        "arm-linux-gnueabihf-gcc",
        "@gcc_linaro_7_3_1_arm_linux_gnueabihf//:gcc",
    ],
)

filegroup(
    name = "ar",
    srcs = [
        "arm-linux-gnueabihf-ar",
        "@gcc_linaro_7_3_1_arm_linux_gnueabihf//:ar",
    ],
)

filegroup(
    name = "ld",
    srcs = [
        "arm-linux-gnueabihf-ld",
        "@gcc_linaro_7_3_1_arm_linux_gnueabihf//:ld",
    ],
)

filegroup(
    name = "nm",
    srcs = [
        "arm-linux-gnueabihf-nm",
        "@gcc_linaro_7_3_1_arm_linux_gnueabihf//:nm",
    ],
)

filegroup(
    name = "objcopy",
    srcs = [
        "arm-linux-gnueabihf-objcopy",
        "@gcc_linaro_7_3_1_arm_linux_gnueabihf//:objcopy",
    ],
)

filegroup(
    name = "objdump",
    srcs = [
        "arm-linux-gnueabihf-objdump",
        "@gcc_linaro_7_3_1_arm_linux_gnueabihf//:objdump",
    ],
)

filegroup(
    name = "strip",
    srcs = [
        "arm-linux-gnueabihf-strip",
        "@gcc_linaro_7_3_1_arm_linux_gnueabihf//:strip",
    ],
)

filegroup(
    name = "as",
    srcs = [
        "arm-linux-gnueabihf-as",
        "@gcc_linaro_7_3_1_arm_linux_gnueabihf//:as",
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
