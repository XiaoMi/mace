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

def if_production_mode(a):
  return select({
      "//mace:production_mode": a,
      "//conditions:default": [],
  })

def if_not_production_mode(a):
  return select({
      "//mace:production_mode": [],
      "//conditions:default": a,
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
