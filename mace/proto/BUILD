# Description:
# mace proto.
#

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("@com_google_protobuf//:protobuf.bzl",
     "py_proto_library",
     "cc_proto_library")

py_proto_library(
    name = "mace_py",
    srcs = ["mace.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY2AND3",
    deps = ["@com_google_protobuf//:protobuf_python"],
)

cc_proto_library(
    name = "mace_cc",
    srcs = ["mace.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_lite",
    protoc = "@com_google_protobuf//:protoc",
)
