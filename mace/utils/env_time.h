// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
