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

#include <utility>
#include <algorithm>
#include <functional>
#include <vector>

#include "mace/core/threadpool.h"
#include "mace/utils/logging.h"

namespace mace {

namespace {
class InitTask : public gemmlowp::Task {
 public:
  explicit InitTask(const std::vector<int> &cpu_ids)
      : Task(), cpu_ids_(cpu_ids) {}
  void Run() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (auto cpu_id : cpu_ids_) {
      CPU_SET(cpu_id, &mask);
    }
    SetThreadAffinity(mask);
  }

 private:
  const std::vector<int> cpu_ids_;
};

class BlockTask : public gemmlowp::Task {
 public:
  BlockTask(const int64_t start,
            const int64_t end,
            const std::function<void(const int64_t, const int64_t)> &fn)
      : Task(),
        start_(start),
        end_(end),
        fn_(fn) {}

  void Run() {
    fn_(start_, end_);
  }

 private:
  const int64_t start_;
  const int64_t end_;
  const std::function<void(int64_t, int64_t)> fn_;
};
}  // namespace

ThreadPool::ThreadPool(int num_threads, CPUAffinityPolicy policy)
    : num_threads_(num_threads), busy_(false) {
  std::vector<int> big_core_ids;
  std::vector<int> little_core_ids;
  std::vector<int> use_cpu_ids;
  MaceStatus res = GetCPUBigLittleCoreIDs(&big_core_ids, &little_core_ids);
  if (res == MaceStatus::MACE_SUCCESS) {
    if (policy == CPUAffinityPolicy::AFFINITY_BIG_ONLY) {
      use_cpu_ids = std::move(big_core_ids);
    } else {
      use_cpu_ids = std::move(little_core_ids);
    }

    if (num_threads <= 0 ||
        num_threads > static_cast<int>(use_cpu_ids.size())) {
      num_threads_ = use_cpu_ids.size();
    }
  } else {
    LOG(WARNING) << "Failed to get cpu big little cores info";
  }

  // if set mask failed and user set num_threads as -1, use single thread
  if (num_threads_ < 0) {
    num_threads_ = 1;
  }

  VLOG(2) << "Use thread count: " << num_threads_;

  std::vector<gemmlowp::Task *> init_tasks(num_threads_);
  for (int i = 0; i < num_threads_; ++i) {
    init_tasks[i] = new InitTask(use_cpu_ids);
  }
  workers_pool_.Execute(init_tasks);
}

ThreadPool::~ThreadPool() {}

void ThreadPool::ParallelRun(const int64_t total,
                             std::function<void(const int64_t,
                                                const int64_t)> fn) {
  if (total <= 0) {
    return;
  } else if (total == 1 || busy_) {
    fn(0, total);
    return;
  }
  busy_ = true;

  const int64_t
      shards = std::min(total, static_cast<const int64_t>(num_threads_));
  const int64_t work_size_per_shard = total / shards;
  const int64_t remain = total - shards * work_size_per_shard;
  const int64_t work_size_per_shard_plus_one = work_size_per_shard + 1;

  std::vector<gemmlowp::Task *> tasks(shards);

  int64_t start = 0;
  int64_t end = 0;
  for (int64_t i = 0; i < remain; ++i) {
    end = start + work_size_per_shard_plus_one;
    tasks[i] = new BlockTask(start, end, fn);
    start = end;
  }
  for (int64_t i = remain; i < shards; ++i) {
    end = start + work_size_per_shard;
    tasks[i] = new BlockTask(start, end, fn);
    start = end;
  }

  workers_pool_.Execute(tasks);
  busy_ = false;
}

void ThreadPool::ParallelFor(const int64_t start,
                             const int64_t end,
                             const int64_t step,
                             std::function<void(const int64_t)> fn) {
  MACE_CHECK(start <= end && step > 0, "start must be le end and step gt 0");
  const int64_t total = (end - start + step - 1) / step;
  ParallelRun(total, [&](const int64_t s, const int64_t t) {
    const int64_t start_i = start + s * step;
    const int64_t end_i = std::min(end, start + t * step);
    for (int64_t i = start_i; i < end_i; i += step) {
      fn(i);
    }
  });
}

void ThreadPool::ParallelFor(const int64_t start1,
                             const int64_t end1,
                             const int64_t step1,
                             const int64_t start2,
                             const int64_t end2,
                             const int64_t step2,
                             std::function<void(const int64_t,
                                                const int64_t)> fn) {
  MACE_CHECK(start1 <= end1 && step1 > 0 && start2 <= end2 && step2 > 0,
             "start must be le end and step gt 0");

  const int64_t total1 = ((end1 - start1 + step1 - 1) / step1);
  const int64_t total2 = ((end2 - start2 + step2 - 1) / step2);
  const int64_t total = total1 * total2;

  if (total == 0) {
    return;
  } else if (total1 == 1) {
    ParallelFor(start2,
                end2,
                step2,
                [&](const int64_t arg2) {
                  fn(start1, arg2);
                });
  } else if (total1 % num_threads_ == 0) {
    ParallelFor(start1,
                end1,
                step1,
                [&](const int64_t arg1) {
                  for (int64_t i = start2; i < end2; i += step2) {
                    fn(arg1, i);
                  }
                });
  } else {
    ParallelRun(total, [&](const int64_t s, const int64_t t) {
      for (int64_t idx = s; idx < t; ++idx) {
        const int64_t i = idx / total2;
        const int64_t j = idx - i * total2;
        fn(start1 + step1 * i, start2 + step2 * j);
      }
    });
  }
}

void ThreadPool::ParallelFor(const int64_t start1,
                             const int64_t end1,
                             const int64_t step1,
                             const int64_t start2,
                             const int64_t end2,
                             const int64_t step2,
                             const int64_t start3,
                             const int64_t end3,
                             const int64_t step3,
                             std::function<void(const int64_t,
                                                const int64_t,
                                                const int64_t)> fn) {
  MACE_CHECK(start1 <= end1 && step1 > 0 && start2 <= end2 && step2 > 0
                 && start3 <= end3 && step3 > 0,
             "start must be le end and step gt 0");

  const int64_t total1 = ((end1 - start1 + step1 - 1) / step1);
  const int64_t total2 = ((end2 - start2 + step2 - 1) / step2);
  const int64_t total3 = ((end3 - start3 + step3 - 1) / step3);
  const int64_t total23 = total2 * total3;
  const int64_t total = total1 * total23;

  if (total == 0) {
    return;
  } else if (total1 == 1) {
    ParallelFor(start2,
                end2,
                step2,
                start3,
                end3,
                step3,
                [&](const int64_t arg2, const int64_t arg3) {
                  fn(start1, arg2, arg3);
                });
  } else if ((total1 * total2) % num_threads_ == 0) {
    ParallelFor(start1,
                end1,
                step1,
                start2,
                end2,
                step2,
                [&](const int64_t arg1, const int64_t arg2) {
                  for (int64_t i = start3; i < end3; i += step3) {
                    fn(arg1, arg2, i);
                  }
                });
  } else {
    ParallelRun(total, [&](int64_t s, int64_t t) {
      for (int64_t idx = s; idx < t; ++idx) {
        const int64_t i = idx / total23;
        const int64_t mod_i = idx - i * total23;
        const int64_t j = mod_i / total3;
        const int64_t k = mod_i - j * total3;
        fn(start1 + step1 * i, start2 + step2 * j, start3 + step3 * k);
      }
    });
  }
}

ThreadPool *ThreadPoolRegister::thread_pool = nullptr;

ThreadPool *ThreadPoolRegister::GetThreadPool() {
  if (thread_pool == nullptr) {
    ConfigThreadPool(-1, CPUAffinityPolicy::AFFINITY_NONE);
  }
  return thread_pool;
}

void ThreadPoolRegister::ConfigThreadPool(int num_threads,
                                          CPUAffinityPolicy policy) {
  MACE_CHECK(thread_pool == nullptr,
             "ThreadPool has already been initialized.");
  thread_pool = new ThreadPool(num_threads, policy);
}

}  // namespace mace
