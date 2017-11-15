py_library(
    name = "tf_converter_lib",
    srcs = [
        "tf_converter_lib.py",
        "tf_dsp_converter_lib.py",
        "graph_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//mace/proto:mace_py",
    ],
)

py_binary(
    name = "tf_converter",
    srcs = ["tf_converter.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tf_converter_lib",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "tf_ops_stats",
    srcs = ["tf_ops_stats.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@six_archive//:six",
    ],
)
