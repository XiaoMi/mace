# Description:
# Mace utils.
#
package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "command_line_flags",
    srcs = [
        "command_line_flags.cc",
    ],
    hdrs = [
        "command_line_flags.h",
    ],
    copts = ["-std=c++11"],
    deps = [
        "//mace/core",
    ],
)

cc_library(
    name = "utils",
    hdrs = [
        "utils.h",
    ],
    copts = ["-std=c++11"],
)

cc_library(
    name = "tuner",
    hdrs = [
        "tuner.h",
    ],
    copts = ["-std=c++11"],
    deps = [
        "//mace/core",
    ],
)
