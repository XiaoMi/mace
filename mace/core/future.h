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

#ifndef MACE_CORE_FUTURE_H_
#define MACE_CORE_FUTURE_H_

#include <functional>

#include "mace/utils/logging.h"

namespace mace {

struct CallStats;

// Wait the call to finish and get the stats if param is not nullptr
struct StatsFuture {
  std::function<void(CallStats *)> wait_fn = [](CallStats *) {
    LOG(FATAL) << "wait_fn must be properly set";
  };
};

inline void SetFutureDefaultWaitFn(StatsFuture *future) {
  if (future != nullptr) {
    future->wait_fn = [](CallStats * stats) {
      if (stats != nullptr) {
        stats->start_micros = NowMicros();
        stats->end_micros = stats->start_micros;
      }
    };
  }
}

}  // namespace mace

#endif  // MACE_CORE_FUTURE_H_
