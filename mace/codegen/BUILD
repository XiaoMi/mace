# Description:
# Generated model and runtime code.
#
package(
    default_visibility = ["//visibility:public"],
)

load("//mace:mace.bzl", "mace_version_genrule", "encrypt_opencl_kernel_genrule")

cc_library(
    name = "libmodels",
    srcs = glob(["models/*/*.cc"]),
    hdrs = glob(["models/*/*.h"]),
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
    deps = [
        "//mace/core",
    ],
)

genrule(
    name = "generated_models",
    srcs = [
        ":libmodels",
    ],
    outs = ["libgenerated_models.a"],
    cmd = "binary=`echo $(locations :libmodels) | xargs -n 1 | grep libmodels.a`;" +
          "$(STRIP) -x $$binary -o $@",
    visibility = ["//visibility:public"],
)

mace_version_genrule()

encrypt_opencl_kernel_genrule()

cc_library(
    name = "generated_opencl",
    srcs = ["opencl/encrypt_opencl_kernel.cc"],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
)

cc_library(
    name = "generated_version",
    srcs = ["version/version.cc"],
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
)

cc_library(
    name = "generated_mace_engine_factory",
    hdrs = glob(["engine/*.h"]),
    copts = ["-Werror", "-Wextra", "-Wno-missing-field-initializers"],
    deps = [
        "//mace/public",
    ],
)

cc_library(
  name = "generated_libmace",
  srcs = glob(["lib/*"]),
  linkstatic = 1,
  visibility = ["//visibility:public"],
)
