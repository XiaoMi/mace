// Copyright 2020 The MACE Authors. All Rights Reserved.
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



#include "micro/port/api.h"

#include <stdlib.h>
#include <stdio.h>
#ifdef MACE_ENABLE_HEXAGON
#include <HAP_perf.h>
#include <HAP_farf.h>
#else
#include <sys/time.h>
#endif

namespace micro {
namespace port {
namespace api {

void DebugLog(const char *str) {
  // you should rewrite this file in the platform source file.
#ifdef MACE_ENABLE_HEXAGON
  FARF(ALWAYS, "%s", str);
#else
  printf("%s", str);
#endif
}

int64_t NowMicros() {
  // you should rewrite this file in the platform source file.
#ifdef MACE_ENABLE_HEXAGON
  return HAP_perf_get_time_us();
#else
  struct timeval tv;
  gettimeofday(&tv, 0);
  return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#endif
}

void Abort() {
  // you should rewrite this file in the platform source file.
  abort();
}

}  // namespace api
}  // namespace port
}  // namespace micro
