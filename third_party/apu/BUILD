licenses(["notice"])

exports_files(["license.txt"])

load(
    "//mace:mace.bzl",
    "if_apu_ancient",
)

cc_library(
    name = "libapu-frontend",
    srcs = if_apu_ancient(
        ["android_Q/mt67xx/libapu-frontend.so"],
        # for mt68xx on android Q the so file is a place holder
        ["android_R/libapu-frontend.so"],
    ),
    hdrs = if_apu_ancient(
        ["android_Q/mt67xx/ApuFrontend.h"],
        # for mt68xx on android Q the header file is a place holder
        ["ApuFrontend.h"],
    ),
    copts = [
        "-DNN_TARGET_NDK",
    ] + if_apu_ancient([
        "-DMACE_MTK_APU_ANCIENT",
    ]),
    linkopts = ["-Wl,-unresolved-symbols=ignore-in-shared-libs"],
    visibility = ["//visibility:public"],
)
