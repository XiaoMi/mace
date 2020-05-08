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

#include "mace/core/runtime/cpu/cpu_runtime.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "mace/port/env.h"
#include "mace/public/mace.h"
#include "mace/utils/macros.h"
#include "mace/utils/logging.h"
#include "mace/utils/thread_pool.h"

namespace mace {

MaceStatus CPURuntime::SetThreadsHintAndAffinityPolicy(
    int num_threads_hint,
    CPUAffinityPolicy policy,
    void *gemm_context) {
  // get cpu frequency info
  std::vector<float> cpu_max_freqs;
  MACE_RETURN_IF_ERROR(GetCPUMaxFreq(&cpu_max_freqs));
  if (cpu_max_freqs.empty()) {
    return MaceStatus::MACE_RUNTIME_ERROR;
  }

  std::vector<size_t> cores_to_use;
  MACE_RETURN_IF_ERROR(
      mace::utils::GetCPUCoresToUse(
          cpu_max_freqs, policy, &num_threads_hint, &cores_to_use));

  if (policy == CPUAffinityPolicy::AFFINITY_NONE) {
#ifdef MACE_ENABLE_QUANTIZE
    if (gemm_context) {
      static_cast<gemmlowp::GemmContext*>(gemm_context)->set_max_num_threads(
          num_threads_hint);
    }
#else
    MACE_UNUSED(gemm_context);
#endif  // MACE_ENABLE_QUANTIZE

    return MaceStatus::MACE_SUCCESS;
  }

#ifdef MACE_ENABLE_QUANTIZE
  if (gemm_context) {
    static_cast<gemmlowp::GemmContext*>(gemm_context)->set_max_num_threads(
        num_threads_hint);
  }
#endif  // MACE_ENABLE_QUANTIZE

  MaceStatus status = SchedSetAffinity(cores_to_use);
  VLOG(1) << "Set affinity : " << MakeString(cores_to_use);

  return status;
}

}  // namespace mace

