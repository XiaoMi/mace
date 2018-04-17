//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_UTILS_ENV_TIME_H_
#define MACE_UTILS_ENV_TIME_H_

#include <stdint.h>
#ifdef __hexagon__
#include <HAP_perf.h>
#else
#include <sys/time.h>
#endif

namespace mace {

inline int64_t NowMicros() {
#ifdef __hexagon__
  return HAP_perf_get_time_us();
#else
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#endif
}

}  // namespace mace

#endif  // MACE_UTILS_ENV_TIME_H_
