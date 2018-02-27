//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_UTILS_ENV_TIME_H
#define MACE_UTILS_ENV_TIME_H

#include <stdint.h>
#include <sys/time.h>
#include <time.h>


namespace mace {

inline int64_t NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

}  // namespace mace

#endif  // MACE_UTILS_ENV_TIME_H
