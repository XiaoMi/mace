# Description:
# Borrow from tensorflow
# Exports generated files used to generate mace/codegen/version/version.cc

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(
    glob(["gen/*"]),
)