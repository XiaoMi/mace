// Copyright 2019 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_UTILS_THREAD_POOL_H_
#define MACE_UTILS_THREAD_POOL_H_

#include <functional>
#include <condition_variable>  // NOLINT(build/c++11)
#include <mutex>  // NOLINT(build/c++11)
#include <thread>  // NOLINT(build/c++11)
#include <vector>
#include <atomic>

#include "mace/public/mace.h"
#include "mace/port/port.h"
#include "mace/utils/count_down_latch.h"

namespace mace {
namespace utils {

class ThreadPool {
 public:
  ThreadPool(const size_t thread_count,
             const CPUAffinityPolicy affinity_policy);
  ~ThreadPool();

  void Init();

  void Run(const std::function<void(size_t)> &func, size_t iterations);

  void Compute1D(const std::function<void(size_t /* start */,
                                          size_t /* end */,
                                          size_t /* step */)> &func,
                 size_t start,
                 size_t end,
                 size_t step,
                 size_t tile_size = 0,
                 int cost_per_item = -1);

  void Compute2D(const std::function<void(size_t /* start */,
                                          size_t /* end */,
                                          size_t /* step */,
                                          size_t /* start */,
                                          size_t /* end */,
                                          size_t /* step */)> &func,
                 size_t start0,
                 size_t end0,
                 size_t step0,
                 size_t start1,
                 size_t end1,
                 size_t step1,
                 size_t tile_size0 = 0,
                 size_t tile_size1 = 0,
                 int cost_per_item = -1);

  void Compute3D(const std::function<void(size_t /* start */,
                                          size_t /* end */,
                                          size_t /* step */,
                                          size_t /* start */,
                                          size_t /* end */,
                                          size_t /* step */,
                                          size_t /* start */,
                                          size_t /* end */,
                                          size_t /* step */)> &func,
                 size_t start0,
                 size_t end0,
                 size_t step0,
                 size_t start1,
                 size_t end1,
                 size_t step1,
                 size_t start2,
                 size_t end2,
                 size_t step2,
                 size_t tile_size0 = 0,
                 size_t tile_size1 = 0,
                 size_t tile_size2 = 0,
                 int cost_per_item = -1);

 private:
  void Destroy();
  void ThreadLoop(size_t tid);
  void ThreadRun(size_t tid);

  std::atomic<int> event_;
  CountDownLatch count_down_latch_;

  std::mutex event_mutex_;
  std::condition_variable event_cond_;
  std::mutex run_mutex_;

  struct ThreadInfo {
    size_t range_start;
    std::atomic<size_t> range_end;
    std::atomic<size_t> range_len;
    uintptr_t func;
    std::vector<size_t> cpu_cores;
  };
  std::vector<ThreadInfo> thread_infos_;
  std::vector<std::thread> threads_;

  size_t default_tile_count_;
};

}  // namespace utils
}  // namespace mace

#endif  // MACE_UTILS_THREAD_POOL_H_
