//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_PUBLIC_VERSION_H_
#define MACE_CORE_PUBLIC_VERSION_H_

#define MACE_MAJOR_VERSION 0
#define MACE_MINOR_VERSION 1
#define MACE_PATCH_VERSION 0

// MACE_VERSION_SUFFIX is non-empty for pre-releases (e.g. "-alpha", "-alpha.1",
// "-beta", "-rc", "-rc.1")
#define MACE_VERSION_SUFFIX ""

#define MACE_STR_HELPER(x) #x
#define MACE_STR(x) MACE_STR_HELPER(x)

// e.g. "0.5.0" or "0.6.0-alpha".
#define MACE_VERSION_STRING                                            \
  (MACE_STR(MACE_MAJOR_VERSION) "." MACE_STR(MACE_MINOR_VERSION) "." MACE_STR( \
      MACE_PATCH_VERSION) MACE_VERSION_SUFFIX)

extern const char *MaceVersion();
extern const char *MaceGitVersion();
#endif //  MACE_CORE_PUBLIC_VERSION_H_
