licenses(["notice"])

exports_files(["license.txt"])

load(
    "//mace:mace.bzl",
    "apu_version_select",
)

cc_library(
    name = "libapu-frontend",
    srcs = apu_version_select(
        # For mt67xx on android Q
        ["android_Q/mt67xx/libapu-frontend.so"],
        # For mt68xx on android Q
        ["android_Q/mt68xx/libapu-frontend.so"],
        # For android R except mt689x and mt6877
        ["android_R/libapu-frontend.so"],
        # For mt689x and mt6877 on android R, for all on android S
        [],
    ),
    hdrs = apu_version_select(
        # For mt67xx on android Q
        ["android_Q/mt67xx/ApuFrontend.h"],
        # For mt68xx on android Q, the header file is a place holder
        ["android_R/ApuFrontend.h"],
        # For android R except mt689x and mt6877
        ["android_R/ApuFrontend.h"],
        # For mt689x and mt6877 on android R, for all on android S
        ["android_R2/neuron_types.h"],
    ),
    copts = [
        "-DNN_TARGET_NDK",
    ],
    linkopts = ["-Wl,-unresolved-symbols=ignore-in-shared-libs"],
    visibility = ["//visibility:public"],
)
