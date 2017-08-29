# Description:
# mace proto.
#
package(
    default_visibility = ["//visibility:public"],
)

proto_library(
    name = "proto",
    srcs = ["mace.proto"],
)

cc_proto_library(
    name = "cc_proto",
    deps = [":proto"],
)