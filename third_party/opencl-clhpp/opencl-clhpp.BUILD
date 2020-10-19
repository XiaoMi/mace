licenses(["notice"])

exports_files(["LICENSE.txt"])

genrule(
    name = "gen_opencl_clhpp",
    srcs = glob([
      "*",
      "**/*",
    ]),
    outs = ["include/CL/cl.hpp", "include/CL/cl2.hpp"],
    cmd = "workdir=$$(mktemp -d -t opencl-clhpp-build.XXXXXXXXXX); cp -aL $$(dirname $(location CMakeLists.txt))/* $$workdir; pushd $$workdir; mkdir build; pushd build; cmake ../ -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF; make generate_clhpp generate_cl2hpp; popd; popd; cp -a $$workdir/build/* $(@D); rm -rf $$workdir; echo installing to  $(@D)",
)

# The `srcs` is not used in c++ Code, but we need it to trigger the `genrule`,
# So we add the "include/CL/cl.hpp", "include/CL/cl2.hpp" into `srcs`, these
# two files is imported by the `includes` instead of `srcs`.
cc_library(
    name = "opencl_clhpp",
    includes = ["include"],
    srcs = ["include/CL/cl.hpp", "include/CL/cl2.hpp"],
    visibility = ["//visibility:public"],
)
