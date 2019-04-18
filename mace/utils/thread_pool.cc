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

#include <algorithm>
#include <numeric>

#include "mace/port/port.h"
#include "mace/port/env.h"
#include "mace/utils/logging.h"
#include "mace/utils/spinlock.h"
#include "mace/utils/math.h"
#include "mace/utils/thread_pool.h"

namespace mace {
namespace utils {

constexpr int kThreadPoolSpinWaitTime = 2000000;  // ns
constexpr int kTileCountPerThread = 2;
constexpr int kMaxCostUsingSingleThread = 100;
constexpr int kMinCpuCoresForPerformance = 3;
constexpr int kMaxCpuCoresForPerformance = 5;

namespace {

enum {
  kThreadPoolNone = 0,
  kThreadPoolInit = 1,
  kThreadPoolRun = 2,
  kThreadPoolShutdown = 4,
  kThreadPoolEventMask = 0x7fffffff
};

struct CPUFreq {
  size_t core_id;
  float freq;
};

size_t GetCpuCoresForPerfomance(const std::vector<CPUFreq> &cpu_freqs) {
  float total_freq = std::accumulate(cpu_freqs.begin(), cpu_freqs.end(), 0,
                                     [](float accum, CPUFreq cpu_freq) {
                                       return accum + cpu_freq.freq;
                                     });
  size_t valid_cpu_nums = std::count_if(cpu_freqs.begin(), cpu_freqs.end(),
                                        [](CPUFreq cpu_freq) {
                                          return cpu_freq.freq != 0;
                                        });
  float avg_freq = total_freq / valid_cpu_nums;

  size_t cores_to_use = 0;
  for (auto cpu_info : cpu_freqs) {
    if ((cpu_info.freq > avg_freq
        && cores_to_use < kMaxCpuCoresForPerformance)
        || cores_to_use < kMinCpuCoresForPerformance) {
      ++cores_to_use;
    }
  }

  return cores_to_use;
}

}  // namespace

MaceStatus GetCPUCoresToUse(const std::vector<float> &cpu_max_freqs,
                            const CPUAffinityPolicy policy,
                            const size_t thread_count_hint,
                            std::vector<size_t> *cores) {
  if (cpu_max_freqs.empty()) {
    LOG(ERROR) << "CPU core is empty";
    return MaceStatus::MACE_RUNTIME_ERROR;
  }
  size_t thread_count = thread_count_hint;
  const size_t cpu_count = cpu_max_freqs.size();
  if (thread_count == 0 || thread_count > cpu_count) {
    thread_count = cpu_count;
  }

  if (policy != CPUAffinityPolicy::AFFINITY_NONE) {
    std::vector<CPUFreq> cpu_freq(cpu_max_freqs.size());
    for (size_t i = 0; i < cpu_max_freqs.size(); ++i) {
      cpu_freq[i].core_id = i;
      cpu_freq[i].freq = cpu_max_freqs[i];
    }
    if (policy == CPUAffinityPolicy::AFFINITY_POWER_SAVE ||
        policy == CPUAffinityPolicy::AFFINITY_LITTLE_ONLY) {
      std::sort(cpu_freq.begin(),
                cpu_freq.end(),
                [=](const CPUFreq &lhs, const CPUFreq &rhs) {
                  return lhs.freq < rhs.freq;
                });
    } else if (policy == CPUAffinityPolicy::AFFINITY_HIGH_PERFORMANCE ||
        policy == CPUAffinityPolicy::AFFINITY_BIG_ONLY) {
      std::sort(cpu_freq.begin(),
                cpu_freq.end(),
                [](const CPUFreq &lhs, const CPUFreq &rhs) {
                  return lhs.freq > rhs.freq;
                });
    }

    // decide num of cores to use
    size_t cores_to_use = 0;
    if (policy == CPUAffinityPolicy::AFFINITY_BIG_ONLY
        || policy == CPUAffinityPolicy::AFFINITY_LITTLE_ONLY) {
      cores_to_use = GetCpuCoresForPerfomance(cpu_freq);
    } else {
      cores_to_use = thread_count;
    }
    MACE_CHECK(cores_to_use > 0, "number of cores to use should > 0");
    cores->resize(cores_to_use);
    for (size_t i = 0; i < cores_to_use; ++i) {
      VLOG(2) << "Bind thread to core: " << cpu_freq[i].core_id
              << " with freq "
              << cpu_freq[i].freq;
      (*cores)[i] = static_cast<int>(cpu_freq[i].core_id);
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

ThreadPool::ThreadPool(const size_t thread_count_hint,
                       const CPUAffinityPolicy policy)
    : event_(kThreadPoolNone),
      count_down_latch_(kThreadPoolSpinWaitTime) {
  size_t thread_count = thread_count_hint;

  std::vector<float> cpu_max_freqs;
  if (port::Env::Default()->GetCPUMaxFreq(&cpu_max_freqs)
      != MaceStatus::MACE_SUCCESS) {
    LOG(ERROR) << "Fail to get cpu max frequencies";
  }

  thread_count = std::max(static_cast<size_t>(1),
                          std::min(thread_count, cpu_max_freqs.size()));

  std::vector<size_t> cores_to_use;
  GetCPUCoresToUse(cpu_max_freqs, policy, thread_count, &cores_to_use);
  if (!cores_to_use.empty()) {
    if (port::Env::Default()->SchedSetAffinity(cores_to_use)
        != MaceStatus::MACE_SUCCESS) {
      LOG(ERROR) << "Failed to sched_set_affinity";
    }
  }
  if (!cores_to_use.empty() && thread_count > cores_to_use.size()) {
    thread_count = cores_to_use.size();
  }
  VLOG(2) << "Use " << thread_count << " threads";

  default_tile_count_ = thread_count;
  if (cores_to_use.size() >= 2
      && cpu_max_freqs[cores_to_use[0]] != cpu_max_freqs[cores_to_use.back()]) {
    default_tile_count_ = thread_count * kTileCountPerThread;
  }
  MACE_CHECK(default_tile_count_ > 0, "default tile count should > 0");

  threads_ = std::vector<std::thread>(thread_count);
  thread_infos_ = std::vector<ThreadInfo>(thread_count);
  for (auto &thread_info : thread_infos_) {
    thread_info.cpu_cores = cores_to_use;
  }
}

ThreadPool::~ThreadPool() {
  Destroy();
}

void ThreadPool::Init() {
  VLOG(2) << "Init thread pool";
  if (threads_.size() <= 1) {
    return;
  }
  count_down_latch_.Reset(static_cast<int>(threads_.size() - 1));
  event_ = kThreadPoolInit;
  for (size_t i = 1; i < threads_.size(); ++i) {
    threads_[i] = std::thread(&ThreadPool::ThreadLoop, this, i);
  }
  count_down_latch_.Wait();
}

void ThreadPool::Run(const std::function<void(const int64_t)> &func,
                     const int64_t iterations) {
  const size_t thread_count = threads_.size();
  const int64_t iters_per_thread = iterations / thread_count;
  const int64_t remainder = iterations % thread_count;
  int64_t iters_offset = 0;

  std::unique_lock<std::mutex> run_lock(run_mutex_);

  for (size_t i = 0; i < thread_count; ++i) {
    int64_t range_len =
        iters_per_thread + (static_cast<int64_t>(i) < remainder);
    thread_infos_[i].range_start = iters_offset;
    thread_infos_[i].range_len = range_len;
    thread_infos_[i].range_end = iters_offset + range_len;
    thread_infos_[i].func = reinterpret_cast<uintptr_t>(&func);
    iters_offset = thread_infos_[i].range_end;
  }

  count_down_latch_.Reset(static_cast<int>(thread_count - 1));
  {
    std::unique_lock<std::mutex> m(event_mutex_);
    event_.store(kThreadPoolRun | ~(event_ | kThreadPoolEventMask),
                 std::memory_order::memory_order_release);
    event_cond_.notify_all();
  }

  ThreadRun(0);
  count_down_latch_.Wait();
}

void ThreadPool::Destroy() {
  VLOG(2) << "Destroy thread pool";
  if (threads_.size() <= 1) {
    return;
  }

  std::unique_lock<std::mutex> run_lock(run_mutex_);

  count_down_latch_.Wait();
  {
    std::unique_lock<std::mutex> m(event_mutex_);
    event_.store(kThreadPoolShutdown, std::memory_order::memory_order_release);
    event_cond_.notify_all();
  }

  for (size_t i = 1; i < threads_.size(); ++i) {
    if (threads_[i].joinable()) {
      threads_[i].join();
    } else {
      VLOG(2) << "Thread: " << threads_[i].get_id() << " not joinable"
              << std::endl;
    }
  }
}

// Event is executed synchronously.
void ThreadPool::ThreadLoop(size_t tid) {
  if (!thread_infos_[tid].cpu_cores.empty()) {
    if (port::Env::Default()->SchedSetAffinity(thread_infos_[tid].cpu_cores)
        != MaceStatus::MACE_SUCCESS) {
      LOG(ERROR) << "Failed to sched set affinity for tid: " << tid;
    }
  }

  int last_event = kThreadPoolNone;

  for (;;) {
    SpinWait(event_, last_event, kThreadPoolSpinWaitTime);
    if (event_.load(std::memory_order::memory_order_acquire) == last_event) {
      std::unique_lock<std::mutex> m(event_mutex_);
      while (event_ == last_event) {
        event_cond_.wait(m);
      }
    }

    int event = event_.load(std::memory_order::memory_order_acquire);
    switch (event & kThreadPoolEventMask) {
      case kThreadPoolInit: {
        count_down_latch_.CountDown();
        break;
      }

      case kThreadPoolRun: {
        ThreadRun(tid);
        count_down_latch_.CountDown();
        break;
      }

      case kThreadPoolShutdown: return;
      default: break;
    }

    last_event = event;
  }
}

void ThreadPool::ThreadRun(size_t tid) {
  ThreadInfo &thread_info = thread_infos_[tid];
  uintptr_t func_ptr = thread_info.func;
  const std::function<void(int64_t)> *func =
      reinterpret_cast<const std::function<void(int64_t)> *>(func_ptr);
  // do own work
  int64_t range_len;
  while ((range_len = thread_info.range_len) > 0) {
    if (thread_info.range_len.compare_exchange_strong(range_len,
                                                      range_len - 1)) {
      func->operator()(thread_info.range_start++);
    }
  }

  // steal other threads' work
  size_t thread_count = threads_.size();
  for (size_t t = (tid + 1) % thread_count; t != tid;
       t = (t + 1) % thread_count) {
    ThreadInfo &other_thread_info = thread_infos_[t];
    uintptr_t other_func_ptr = other_thread_info.func;
    const std::function<void(int64_t)> *other_func =
        reinterpret_cast<const std::function<void(int64_t)> *>(
            other_func_ptr);
    while ((range_len = other_thread_info.range_len) > 0) {
      if (other_thread_info.range_len.compare_exchange_strong(range_len,
                                                              range_len
                                                                  - 1)) {
        int64_t tail = other_thread_info.range_end--;
        other_func->operator()(tail - 1);
      }
    }
  }
}

void ThreadPool::Compute1D(const std::function<void(int64_t,
                                                    int64_t,
                                                    int64_t)> &func,
                           const int64_t start,
                           const int64_t end,
                           const int64_t step,
                           int64_t tile_size,
                           const int cost_per_item) {
  if (start >= end) {
    return;
  }

  const int64_t items = 1 + (end - start - 1) / step;
  if (threads_.size() <= 1 || (cost_per_item >= 0
      && items * cost_per_item < kMaxCostUsingSingleThread)) {
    func(start, end, step);
    return;
  }

  if (tile_size == 0) {
    tile_size = std::max(static_cast<int64_t>(1), items / default_tile_count_);
  }

  const int64_t step_tile_size = step * tile_size;
  const int64_t tile_count = RoundUpDiv(items, tile_size);
  Run([=](int64_t tile_idx) {
    const int64_t tile_start = start + tile_idx * step_tile_size;
    const int64_t tile_end = std::min(end, tile_start + step_tile_size);
    func(tile_start, tile_end, step);
  }, tile_count);
}

void ThreadPool::Compute2D(const std::function<void(const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t)> &func,
                           const int64_t start0,
                           const int64_t end0,
                           const int64_t step0,
                           const int64_t start1,
                           const int64_t end1,
                           const int64_t step1,
                           int64_t tile_size0,
                           int64_t tile_size1,
                           const int cost_per_item) {
  if (start0 >= end0 || start1 >= end1) {
    return;
  }

  const int64_t items0 = 1 + (end0 - start0 - 1) / step0;
  const int64_t items1 = 1 + (end1 - start1 - 1) / step1;
  if (threads_.size() <= 1 || (cost_per_item >= 0
      && items0 * items1 * cost_per_item < kMaxCostUsingSingleThread)) {
    func(start0, end0, step0, start1, end1, step1);
    return;
  }

  if (tile_size0 == 0 || tile_size1 == 0) {
    if (items0 >= default_tile_count_) {
      tile_size0 = items0 / default_tile_count_;
      tile_size1 = items1;
    } else {
      tile_size0 = 1;
      tile_size1 = std::max(static_cast<int64_t>(1),
                            items1 * items0 / default_tile_count_);
    }
  }

  const int64_t step_tile_size0 = step0 * tile_size0;
  const int64_t step_tile_size1 = step1 * tile_size1;
  const int64_t tile_count0 = RoundUpDiv(items0, tile_size0);
  const int64_t tile_count1 = RoundUpDiv(items1, tile_size1);

  Run([=](int64_t tile_idx) {
    const int64_t tile_idx0 = tile_idx / tile_count1;
    const int64_t tile_idx1 = tile_idx - tile_idx0 * tile_count1;
    const int64_t tile_start0 = start0 + tile_idx0 * step_tile_size0;
    const int64_t tile_end0 = std::min(end0, tile_start0 + step_tile_size0);
    const int64_t tile_start1 = start1 + tile_idx1 * step_tile_size1;
    const int64_t tile_end1 = std::min(end1, tile_start1 + step_tile_size1);
    func(tile_start0, tile_end0, step0, tile_start1, tile_end1, step1);
  }, tile_count0 * tile_count1);
}

void ThreadPool::Compute3D(const std::function<void(const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t)> &func,
                           const int64_t start0,
                           const int64_t end0,
                           const int64_t step0,
                           const int64_t start1,
                           const int64_t end1,
                           const int64_t step1,
                           const int64_t start2,
                           const int64_t end2,
                           const int64_t step2,
                           int64_t tile_size0,
                           int64_t tile_size1,
                           int64_t tile_size2,
                           const int cost_per_item) {
  if (start0 >= end0 || start1 >= end1 || start2 >= end1) {
    return;
  }

  const int64_t items0 = 1 + (end0 - start0 - 1) / step0;
  const int64_t items1 = 1 + (end1 - start1 - 1) / step1;
  const int64_t items2 = 1 + (end2 - start2 - 1) / step2;
  if (threads_.size() <= 1 || (cost_per_item >= 0
      && items0 * items1 * items2 * cost_per_item
          < kMaxCostUsingSingleThread)) {
    func(start0, end0, step0, start1, end1, step1, start2, end2, step2);
    return;
  }

  if (tile_size0 == 0 || tile_size1 == 0 || tile_size2 == 0) {
    if (items0 >= default_tile_count_) {
      tile_size0 = items0 / default_tile_count_;
      tile_size1 = items1;
      tile_size2 = items2;
    } else {
      tile_size0 = 1;
      const int64_t items01 = items1 * items0;
      if (items01 >= default_tile_count_) {
        tile_size1 = items01 / default_tile_count_;
        tile_size2 = items2;
      } else {
        tile_size1 = 1;
        tile_size2 = std::max(static_cast<int64_t>(1),
                              items01 * items2 / default_tile_count_);
      }
    }
  }

  const int64_t step_tile_size0 = step0 * tile_size0;
  const int64_t step_tile_size1 = step1 * tile_size1;
  const int64_t step_tile_size2 = step2 * tile_size2;
  const int64_t tile_count0 = RoundUpDiv(items0, tile_size0);
  const int64_t tile_count1 = RoundUpDiv(items1, tile_size1);
  const int64_t tile_count2 = RoundUpDiv(items2, tile_size2);
  const int64_t tile_count12 = tile_count1 * tile_count2;

  Run([=](int64_t tile_idx) {
    const int64_t tile_idx0 = tile_idx / tile_count12;
    const int64_t tile_idx12 = tile_idx - tile_idx0 * tile_count12;
    const int64_t tile_idx1 = tile_idx12 / tile_count2;
    const int64_t tile_idx2 = tile_idx12 - tile_idx1 * tile_count2;
    const int64_t tile_start0 = start0 + tile_idx0 * step_tile_size0;
    const int64_t tile_end0 = std::min(end0, tile_start0 + step_tile_size0);
    const int64_t tile_start1 = start1 + tile_idx1 * step_tile_size1;
    const int64_t tile_end1 = std::min(end1, tile_start1 + step_tile_size1);
    const int64_t tile_start2 = start2 + tile_idx2 * step_tile_size2;
    const int64_t tile_end2 = std::min(end2, tile_start2 + step_tile_size2);
    func(tile_start0,
         tile_end0,
         step0,
         tile_start1,
         tile_end1,
         step1,
         tile_start2,
         tile_end2,
         step2);
  }, tile_count0 * tile_count12);
}

}  // namespace utils
}  // namespace mace
