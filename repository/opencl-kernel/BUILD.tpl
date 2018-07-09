# Description:
# Exports generated files used to generate mace/codegen/opencl/opencl_encrypt_program.cc

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(
    glob(["gen/*"]),
)