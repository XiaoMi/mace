py_library(
    name = "tf_converter_lib",
    srcs = [
        "convert_util.py",
        "graph_util.py",
        "tf_converter_lib.py",
        "tf_dsp_converter_lib.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":memory_optimizer",
        "//lib/proto:mace_py",
    ],
)

py_library(
    name = "source_converter_lib",
    srcs = [
        "source_converter_lib.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//lib/proto:mace_py",
    ],
)

py_binary(
    name = "tf_converter",
    srcs = ["tf_converter.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tf_converter_lib",
        ":source_converter_lib",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "memory_optimizer",
    srcs = ["memory_optimizer.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//lib/proto:mace_py",
    ],
)
