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

#ifndef MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_
#define MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_

#include <memory>
#include <vector>

#include "public/gemmlowp.h"
#include "mace/public/mace.h"
#include "mace/utils/logging.h"

namespace mace {

extern int MaceOpenMPThreadCount;

class CPURuntime {
 public:
  CPURuntime(const int num_threads,
             CPUAffinityPolicy policy,
             bool use_gemmlowp)
      : num_threads_(num_threads),
        policy_(policy),
        gemm_context_(nullptr) {
    if (use_gemmlowp) {
      MACE_CHECK_NOTNULL(GetGemmlowpContext());
    }

    SetOpenMPThreadsAndAffinityPolicy(num_threads_,
                                      policy_,
                                      gemm_context_.get());
  }
  ~CPURuntime() = default;

  gemmlowp::GemmContext *GetGemmlowpContext() {
    if (!gemm_context_) {
      gemm_context_.reset(new gemmlowp::GemmContext());
    }
    return gemm_context_.get();
  }

  int num_threads() const {
    return num_threads_;
  }

 private:
  MaceStatus SetOpenMPThreadsAndAffinityPolicy(
      int omp_num_threads_hint,
      CPUAffinityPolicy policy,
      gemmlowp::GemmContext *gemm_context);

  int num_threads_;
  CPUAffinityPolicy policy_;
  std::unique_ptr<gemmlowp::GemmContext> gemm_context_;
};
}  // namespace mace

#endif  // MACE_CORE_RUNTIME_CPU_CPU_RUNTIME_H_
