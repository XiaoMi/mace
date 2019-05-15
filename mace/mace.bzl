# -*- Python -*-

def if_linux_base(a, default_value = []):
  return select({
      "//mace:linux_base": a,
      "//conditions:default": default_value,
  })

def if_android(a, default_value = []):
  return select({
      "//mace:android": a,
      "//conditions:default": default_value,
  })

def if_linux(a, default_value = []):
  return select({
      "//mace:linux": a,
      "//conditions:default": default_value,
  })

def if_darwin(a, default_value = []):
  return select({
      "//mace:darwin": a,
      "//conditions:default": default_value,
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

def if_arm_linux_aarch64(a):
  return select({
      "//mace:arm_linux_aarch64": a,
      "//conditions:default": [],
  })

def if_arm_linux_armhf(a):
  return select({
      "//mace:arm_linux_armhf": a,
      "//conditions:default": []
  })

def if_neon_enabled(a, default_value = []):
  return select({
      "//mace:neon_enabled": a,
      "//conditions:default": default_value,
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

def if_hta_enabled(a):
  return select({
      "//mace:hta_enabled": a,
      "//conditions:default": [],
  })

def if_hexagon_or_hta_enabled(a):
  return select({
      "//mace:hexagon_enabled": a,
      "//mace:hta_enabled": a,
      "//conditions:default": [],
  })

def if_apu_enabled(a):
  return select({
      "//mace:apu_enabled": a,
      "//conditions:default": [],
  })

def if_not_apu_enabled(a):
  return select({
      "//mace:apu_enabled": [],
      "//conditions:default": a,
  })

def if_openmp_enabled(a):
  return select({
      "//mace:openmp_enabled": a,
      "//conditions:default": [],
  })

def if_opencl_enabled(a, default_value = []):
  return select({
      "//mace:opencl_enabled": a,
      "//conditions:default": default_value,
  })

def if_quantize_enabled(a):
  return select({
      "//mace:quantize_enabled": a,
      "//conditions:default": [],
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
