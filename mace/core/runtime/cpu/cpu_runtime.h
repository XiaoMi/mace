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

#ifndef MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_
#define MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_

#include <memory>
#include <vector>

#ifdef MACE_ENABLE_QUANTIZE
#include "public/gemmlowp.h"
#endif  // MACE_ENABLE_QUANTIZE

#include "mace/utils/thread_pool.h"
#include "mace/utils/macros.h"
#include "mace/public/mace.h"
#include "mace/utils/logging.h"

namespace mace {

extern int MaceOpenMPThreadCount;

class CPURuntime {
 public:
  CPURuntime(const int num_threads,
             CPUAffinityPolicy policy,
             utils::ThreadPool *thread_pool)
      : num_threads_(num_threads),
        policy_(policy),
        gemm_context_(nullptr),
        thread_pool_(thread_pool) {
#ifdef MACE_ENABLE_QUANTIZE
    MACE_CHECK_NOTNULL(GetGemmlowpContext());
#endif  // MACE_ENABLE_QUANTIZE
    SetOpenMPThreadsAndAffinityPolicy(num_threads_,
                                      policy_,
                                      gemm_context_);
  }

#ifdef MACE_ENABLE_QUANTIZE
  ~CPURuntime() {
    if (gemm_context_ != nullptr) {
      delete static_cast<gemmlowp::GemmContext*>(gemm_context_);
    }
  }

  gemmlowp::GemmContext *GetGemmlowpContext() {
    if (gemm_context_ == nullptr) {
      gemm_context_ = new gemmlowp::GemmContext(thread_pool_);
    }
    return static_cast<gemmlowp::GemmContext*>(gemm_context_);
  }
#else
  ~CPURuntime() = default;
#endif  // MACE_ENABLE_QUANTIZE

  int num_threads() const {
    return num_threads_;
  }

  CPUAffinityPolicy policy() const {
    return policy_;
  }

  utils::ThreadPool &thread_pool() {
    return *thread_pool_;
  }

 private:
  MaceStatus SetOpenMPThreadsAndAffinityPolicy(
      int omp_num_threads_hint,
      CPUAffinityPolicy policy,
      void *gemm_context);

  int num_threads_;
  CPUAffinityPolicy policy_;
  void *gemm_context_;
  utils::ThreadPool *thread_pool_;
};
}  // namespace mace

#endif  // MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_
