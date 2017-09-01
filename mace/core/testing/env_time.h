//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

// Only support POSIX environment
#ifndef MACE_TESTING_TIME_H_
#define MACE_TESTING_TIME_H_

#include <stdint.h>
#include <sys/time.h>
#include <time.h>

#include "mace/core/types.h"

namespace mace {

namespace testing {

inline int64 NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int64>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

}  // namespace testing
}  // namespace mace

#endif  // MACE_TESTING_TIME_H_
