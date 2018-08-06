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

#include "mace/kernels/gemmlowp_util.h"

#include <algorithm>
#include <vector>

#include "mace/core/runtime/cpu/cpu_runtime.h"

namespace mace {

gemmlowp::GemmContext& GetGemmlowpContext() {
  static auto *gemm_context = new gemmlowp::GemmContext;
  return *gemm_context;
}

MaceStatus SetGemmlowpThreadPolicy(int num_threads_hint,
                                   CPUAffinityPolicy policy) {
  gemmlowp::GemmContext& gemm_context = GetGemmlowpContext();

  if (policy != AFFINITY_NONE) {
    std::vector<int> big_core_ids;
    std::vector<int> little_core_ids;
    MaceStatus res = GetCPUBigLittleCoreIDs(&big_core_ids, &little_core_ids);
    if (res != MACE_SUCCESS) {
      return res;
    }

    int use_cpu_size;
    if (policy == CPUAffinityPolicy::AFFINITY_BIG_ONLY) {
      use_cpu_size = static_cast<int>(big_core_ids.size());
    } else {
      use_cpu_size = static_cast<int>(little_core_ids.size());
    }

    if (num_threads_hint <= 0 || num_threads_hint > use_cpu_size) {
      num_threads_hint = use_cpu_size;
    }
  }

  gemm_context.set_max_num_threads(std::max(0, num_threads_hint));

  return MACE_SUCCESS;
}

}  // namespace mace
