// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include <algorithm>
#include <functional>
#include <vector>

#include "mace/utils/logging.h"
#include "mace/public/mace.h"

namespace mace {

// Wait the call to finish and get the stats if param is not nullptr
struct StatsFuture {
  std::function<void(CallStats *)> wait_fn = [](CallStats *stats) {
    if (stats != nullptr) {
      stats->start_micros = NowMicros();
      stats->end_micros = stats->start_micros;
    }
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

inline void MergeMultipleFutureWaitFn(
    const std::vector<StatsFuture> &org_futures,
    StatsFuture *dst_future) {
  if (dst_future != nullptr) {
    dst_future->wait_fn = [org_futures](CallStats *stats) {
      if (stats != nullptr) {
        stats->start_micros = INT64_MAX;
        stats->end_micros = 0;
        for (auto &org_future : org_futures) {
          CallStats tmp_stats;
          if (org_future.wait_fn != nullptr) {
            org_future.wait_fn(&tmp_stats);
            stats->start_micros = std::min(stats->start_micros,
                                           tmp_stats.start_micros);
            stats->end_micros += tmp_stats.end_micros - tmp_stats.start_micros;
          }
        }
        stats->end_micros += stats->start_micros;
      }
    };
  }
}

}  // namespace mace

#endif  // MACE_CORE_FUTURE_H_
