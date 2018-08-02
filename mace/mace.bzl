# -*- Python -*-

def if_android(a):
  return select({
      "//mace:android": a,
      "//conditions:default": [],
  })

def if_not_android(a):
  return select({
      "//mace:android": [],
      "//conditions:default": a,
  })

def if_android_armv7(a):
  return select({
      "//mace:android_armv7": a,
      "//conditions:default": [],
  })

def if_android_arm64(a):
  return select({
      "//mace:android_arm64": a,
      "//conditions:default": [],
  })

def if_neon_enabled(a):
  return select({
      "//mace:neon_enabled": a,
      "//conditions:default": [],
  })

def if_hexagon_enabled(a):
  return select({
      "//mace:hexagon_enabled": a,
      "//conditions:default": [],
  })

def if_not_hexagon_enabled(a):
  return select({
      "//mace:hexagon_enabled": [],
      "//conditions:default": a,
  })

def if_openmp_enabled(a):
  return select({
      "//mace:openmp_enabled": a,
      "//conditions:default": [],
  })

def if_opencl_enabled(a):
  return select({
      "//mace:opencl_enabled": a,
      "//conditions:default": [],
  })

def if_opencl_enabled_str(a):
  return select({
      "//mace:opencl_enabled": a,
      "//conditions:default": "",
  })

def mace_version_genrule():
  native.genrule(
      name = "mace_version_gen",
      srcs = [str(Label("@local_version_config//:gen/version"))],
      outs = ["version/version.cc"],
      cmd = "cat $(SRCS) > $@;"
  )

def encrypt_opencl_kernel_genrule():
  native.genrule(
      name = "encrypt_opencl_kernel_gen",
      srcs = [str(Label("@local_opencl_kernel_encrypt//:gen/encrypt_opencl_kernel"))],
      outs = ["opencl/encrypt_opencl_kernel.cc"],
      cmd = "cat $(SRCS) > $@;"
  )

