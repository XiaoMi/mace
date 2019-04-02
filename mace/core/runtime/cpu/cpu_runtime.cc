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

#ifdef MACE_ENABLE_OPENMP
#include <omp.h>
#endif

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

namespace mace {

int MaceOpenMPThreadCount = 1;

struct CPUFreq {
  size_t core_id;
  float freq;
};

enum SchedulePolicy {
  SCHED_STATIC,
  SCHED_GUIDED,
};

namespace {

MaceStatus SetOpenMPThreadsAndAffinityCPUs(int omp_num_threads,
                                           const std::vector<size_t> &cpu_ids,
                                           SchedulePolicy schedule_policy) {
  MaceOpenMPThreadCount = omp_num_threads;
  SchedSetAffinity(cpu_ids);
#ifdef MACE_ENABLE_OPENMP
  VLOG(1) << "Set OpenMP threads number: " << omp_num_threads
          << ", CPU core IDs: " << MakeString(cpu_ids);
  if (schedule_policy == SCHED_GUIDED) {
    omp_set_schedule(omp_sched_guided, 1);
  } else if (schedule_policy == SCHED_STATIC) {
    omp_set_schedule(omp_sched_static, 0);
  } else {
    LOG(WARNING) << "Unknown schedule policy: " << schedule_policy;
  }

  omp_set_num_threads(omp_num_threads);
#else
  MACE_UNUSED(omp_num_threads);
  MACE_UNUSED(schedule_policy);
  LOG(WARNING) << "Set OpenMP threads number failed: OpenMP not enabled.";
#endif

#ifdef MACE_ENABLE_OPENMP
  std::vector<MaceStatus> status(omp_num_threads,
                                 MaceStatus::MACE_INVALID_ARGS);
#pragma omp parallel for
  for (int i = 0; i < omp_num_threads; ++i) {
    VLOG(1) << "Set affinity for OpenMP thread " << omp_get_thread_num()
            << "/" << omp_get_num_threads();
    status[i] = SchedSetAffinity(cpu_ids);
  }
  for (int i = 0; i < omp_num_threads; ++i) {
    if (status[i] != MaceStatus::MACE_SUCCESS)
      return MaceStatus::MACE_INVALID_ARGS;
  }
  return MaceStatus::MACE_SUCCESS;
#else
  MaceStatus status = SchedSetAffinity(cpu_ids);
  VLOG(1) << "Set affinity without OpenMP: " << MakeString(cpu_ids);
  return status;
#endif
}

}  // namespace

MaceStatus CPURuntime::SetOpenMPThreadsAndAffinityPolicy(
    int num_threads_hint,
    CPUAffinityPolicy policy,
    void *gemm_context) {
  // get cpu frequency info
  std::vector<float> cpu_max_freqs;
  MACE_RETURN_IF_ERROR(GetCPUMaxFreq(&cpu_max_freqs));
  if (cpu_max_freqs.empty()) {
    return MaceStatus::MACE_RUNTIME_ERROR;
  }

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

  int cpu_count = static_cast<int>(cpu_freq.size());
  if (num_threads_hint <= 0 || num_threads_hint > cpu_count) {
    num_threads_hint = cpu_count;
  }

  if (policy == CPUAffinityPolicy::AFFINITY_NONE) {
#ifdef MACE_ENABLE_QUANTIZE
    if (gemm_context) {
      static_cast<gemmlowp::GemmContext*>(gemm_context)->set_max_num_threads(
          num_threads_hint);
    }
#else
    MACE_UNUSED(gemm_context);
#endif  // MACE_ENABLE_QUANTIZE
#ifdef MACE_ENABLE_OPENMP
    omp_set_num_threads(num_threads_hint);
#else
    LOG(WARNING) << "Set OpenMP threads number failed: OpenMP not enabled.";
#endif
    return MaceStatus::MACE_SUCCESS;
  }


  // decide num of cores to use
  int cores_to_use = 0;
  if (policy == CPUAffinityPolicy::AFFINITY_BIG_ONLY
      || policy == CPUAffinityPolicy::AFFINITY_LITTLE_ONLY) {
    for (size_t i = 0; i < cpu_max_freqs.size(); ++i) {
      if (cpu_freq[i].freq != cpu_freq[0].freq) {
        break;
      }
      ++cores_to_use;
    }
    num_threads_hint = std::min(num_threads_hint, cores_to_use);
  } else {
    cores_to_use = num_threads_hint;
  }
  MACE_CHECK(cores_to_use > 0, "number of cores to use should > 0");

  VLOG(2) << "Use " << num_threads_hint << " threads";
  std::vector<size_t> cpu_ids(cores_to_use);
  for (int i = 0; i < cores_to_use; ++i) {
    VLOG(2) << "Bind thread to core: " << cpu_freq[i].core_id << " with freq "
            << cpu_freq[i].freq;
    cpu_ids[i] = cpu_freq[i].core_id;
  }
  SchedulePolicy sched_policy = SCHED_GUIDED;
  if (std::abs(cpu_freq[0].freq - cpu_freq[cores_to_use - 1].freq) < 1e-6) {
    sched_policy = SCHED_STATIC;
  }

#ifdef MACE_ENABLE_QUANTIZE
  if (gemm_context) {
    static_cast<gemmlowp::GemmContext*>(gemm_context)->set_max_num_threads(
        num_threads_hint);
  }
#endif  // MACE_ENABLE_QUANTIZE

  return SetOpenMPThreadsAndAffinityCPUs(num_threads_hint,
                                         cpu_ids,
                                         sched_policy);
}

}  // namespace mace

