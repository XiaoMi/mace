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

cc_library(
    name = "opencl_clhpp",
    hdrs = ["include/CL/cl.hpp", "include/CL/cl2.hpp"],
    visibility = ["//visibility:public"],
)
