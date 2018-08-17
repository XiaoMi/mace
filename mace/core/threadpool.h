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

#ifndef MACE_CORE_THREADPOOL_H_
#define MACE_CORE_THREADPOOL_H_

#include "mace/core/runtime/cpu/cpu_runtime.h"

// Remove the following macros after removing openmp dependency
#ifdef GEMMLOWP_USE_OPENMP
#define MACE_GEMMLOWP_USE_OPENMP
#undef GEMMLOWP_USE_OPENMP
#endif
#include "internal/multi_thread_gemm.h"
#ifdef MACE_GEMMLOWP_USE_OPENMP
#define GEMMLOWP_USE_OPENMP
#undef MACE_GEMMLOWP_USE_OPENMP
#endif

namespace mace {

class ThreadPool {
 public:
  explicit ThreadPool(int num_threads, CPUAffinityPolicy policy);

  ~ThreadPool();

  void ParallelRun(const int64_t total,
                   std::function<void(const int64_t, const int64_t)> fn);

  // Parallel for
  void ParallelFor(const int64_t start, const int64_t end, const int64_t step,
                   std::function<void(const int64_t)> fn);

  // Parallel for collapse(2)
  void ParallelFor(const int64_t start1,
                   const int64_t end1,
                   const int64_t step1,
                   const int64_t start2,
                   const int64_t end2,
                   const int64_t step2,
                   std::function<void(const int64_t, const int64_t)> fn);

  // Parallel for collapse(3)
  void ParallelFor(const int64_t start1,
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
                                      const int64_t)> fn);

 private:
  int num_threads_;
  bool busy_;
  gemmlowp::WorkersPool workers_pool_;
};

class ThreadPoolRegister {
 public:
  static ThreadPool *GetThreadPool();

  static void ConfigThreadPool(int num_threads, CPUAffinityPolicy policy);

 private:
  static ThreadPool *thread_pool;
};

}  // namespace mace



#endif  // MACE_CORE_THREADPOOL_H_
