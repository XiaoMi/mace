# This is the entry point for --crosstool_top.  Toolchains are found
# by lopping off the name of --crosstool_top and searching for
# 'cc-compiler-${CPU}' in this BUILD file, where CPU is the target CPU
# specified in --cpu.
#
# This file group should include
#   * all cc_toolchain targets supported
#   * all file groups that said cc_toolchain might refer to,
# including the default_grte_top setting in the CROSSTOOL
# protobuf.
filegroup(
    name = "toolchain_fg",
    srcs = [
        ":cc-compiler-armhf",
        ":linaro_linux_all_files",
        "@gcc_linaro_7_3_1_arm_linux_gnueabihf//:compiler_components",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/arm_compiler/linaro_linux_gcc:srcs",
    ],
    visibility = ["//visibility:public"],
)

cc_toolchain_suite(
    name = "toolchain",
    # target_cpu | compiler
    toolchains = {
        "armhf|gcc": "cc-compiler-armhf",
    },
)

filegroup(
    name = "linaro_linux_all_files",
    srcs = [
        "//tools/arm_compiler/linaro_linux_gcc:tool-wrappers",
        "@gcc_linaro_7_3_1_arm_linux_gnueabihf//:compiler_pieces",
    ],
)

filegroup(
    name = "linaro_linux_linker_files",
    srcs = [
        "//tools/arm_compiler/linaro_linux_gcc:ar",
        "//tools/arm_compiler/linaro_linux_gcc:gcc",
        "//tools/arm_compiler/linaro_linux_gcc:ld",
        "@gcc_linaro_7_3_1_arm_linux_gnueabihf//:compiler_pieces",
    ],
)

filegroup(
    name = "linaro_linux_compiler_files",
    srcs = [
        "//tools/arm_compiler/linaro_linux_gcc:as",
        "//tools/arm_compiler/linaro_linux_gcc:gcc",
        "//tools/arm_compiler/linaro_linux_gcc:ld",
    ],
)

filegroup(
    name = "empty",
    srcs = [],
)

cc_toolchain(
    name = "cc-compiler-armhf",
    all_files = ":linaro_linux_all_files",
    compiler_files = ":linaro_linux_compiler_files",
    cpu = "armhf",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":linaro_linux_linker_files",
    objcopy_files = "//tools/arm_compiler/linaro_linux_gcc:objcopy",
    static_runtime_libs = [":empty"],
    strip_files = "//tools/arm_compiler/linaro_linux_gcc:strip",
    supports_param_files = 1,
    visibility = ["//visibility:public"],
)
