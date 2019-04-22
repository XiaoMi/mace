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

MaceStatus GetCPUCoresToUse(const std::vector<float> &cpu_max_freqs,
                            const CPUAffinityPolicy policy,
                            const size_t thread_count_hint,
                            std::vector<size_t> *cores);

class ThreadPool {
 public:
  ThreadPool(const size_t thread_count,
             const CPUAffinityPolicy affinity_policy);
  ~ThreadPool();

  void Init();

  void Run(const std::function<void(const int64_t)> &func,
           const int64_t iterations);

  void Compute1D(const std::function<void(int64_t /* start */,
                                          int64_t /* end */,
                                          int64_t /* step */)> &func,
                 int64_t start,
                 int64_t end,
                 int64_t step,
                 int64_t tile_size = 0,
                 int cost_per_item = -1);

  void Compute2D(const std::function<void(int64_t /* start */,
                                          int64_t /* end */,
                                          int64_t /* step */,
                                          int64_t /* start */,
                                          int64_t /* end */,
                                          int64_t /* step */)> &func,
                 int64_t start0,
                 int64_t end0,
                 int64_t step0,
                 int64_t start1,
                 int64_t end1,
                 int64_t step1,
                 int64_t tile_size0 = 0,
                 int64_t tile_size1 = 0,
                 int cost_per_item = -1);

  void Compute3D(const std::function<void(int64_t /* start */,
                                          int64_t /* end */,
                                          int64_t /* step */,
                                          int64_t /* start */,
                                          int64_t /* end */,
                                          int64_t /* step */,
                                          int64_t /* start */,
                                          int64_t /* end */,
                                          int64_t /* step */)> &func,
                 int64_t start0,
                 int64_t end0,
                 int64_t step0,
                 int64_t start1,
                 int64_t end1,
                 int64_t step1,
                 int64_t start2,
                 int64_t end2,
                 int64_t step2,
                 int64_t tile_size0 = 0,
                 int64_t tile_size1 = 0,
                 int64_t tile_size2 = 0,
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
    std::atomic<int64_t> range_start;
    std::atomic<int64_t> range_end;
    std::atomic<int64_t> range_len;
    uintptr_t func;
    std::vector<size_t> cpu_cores;
  };
  std::vector<ThreadInfo> thread_infos_;
  std::vector<std::thread> threads_;

  int64_t default_tile_count_;
};

}  // namespace utils
}  // namespace mace

#endif  // MACE_UTILS_THREAD_POOL_H_
