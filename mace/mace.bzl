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
