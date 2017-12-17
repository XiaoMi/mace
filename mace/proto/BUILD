# Description:
# mace proto.
#

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

proto_library(
    name = "stats",
    srcs = ["stats.proto"],
)

cc_proto_library(
    name = "stats_proto",
    deps = [":stats"],
)

py_proto_library(
    name = "mace_py",
    srcs = ["mace.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY2AND3",
    deps = ["@com_google_protobuf//:protobuf_python"],
)
