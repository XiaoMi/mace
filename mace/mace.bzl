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

def if_profiling_enabled(a):
  return select({
      "//mace:profiling_enabled": a,
      "//conditions:default": [],
  })

def if_enable_neon(a):
  return select({
      "//mace:enable_neon": a,
      "//conditions:default": [],
  })
