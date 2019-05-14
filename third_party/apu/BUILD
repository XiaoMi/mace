
licenses(["notice"])

exports_files(["license.txt"])

cc_library(
    name = "libapu-frontend",
    srcs = [
        "libapu-frontend.so",
    ],
    hdrs = glob(["*.h"]),
    copts = ["-DNN_TARGET_NDK"],
    linkopts = ["-Wl,-unresolved-symbols=ignore-in-shared-libs"],
    visibility = ["//visibility:public"],
)
