py_library(
    name = "converter_lib",
    srcs = [
        "convert_util.py",
        "graph_util.py",
        "tensor_util.py",
        "tf_dsp_converter_lib.py",
        "converter_tool/base_converter.py",
        "converter_tool/shape_inference.py",
        "converter_tool/tensorflow_converter.py",
        "converter_tool/caffe_converter.py",
        "converter_tool/transformer.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":memory_optimizer",
        "//mace/proto:mace_py",
        "//mace/third_party/caffe:caffe_py",
    ],
)

py_library(
    name = "source_converter_lib",
    srcs = [
        "source_converter_lib.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//mace/proto:mace_py",
    ],
)

py_binary(
    name = "memory_optimizer",
    srcs = ["memory_optimizer.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//mace/proto:mace_py",
    ],
)

py_binary(
    name = "converter",
    srcs = ["converter.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converter_lib",
        ":source_converter_lib",
        "@six_archive//:six",
    ],
)
