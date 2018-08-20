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

#include "mace/core/runtime/cpu/cpu_runtime.h"

#ifdef MACE_ENABLE_OPENMP
#include <omp.h>
#endif

#include <errno.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <string.h>
#include <algorithm>
#include <utility>
#include <vector>

#include "mace/core/macros.h"
#include "mace/public/mace.h"
#include "mace/public/mace_runtime.h"
#include "mace/utils/logging.h"

namespace mace {

namespace {

int GetCPUCount() {
  char path[32];
  int cpu_count = 0;
  int result = 0;

  while (true) {
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d", cpu_count);
    result = access(path, F_OK);
    if (result != 0) {
      if (errno != ENOENT) {
        LOG(ERROR) << "Access " << path << " failed: " << strerror(errno);
      }
      return cpu_count;
    }
    cpu_count++;
  }
}

int GetCPUMaxFreq(int cpu_id) {
  char path[64];
  snprintf(path, sizeof(path),
          "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
          cpu_id);

  FILE *fp = fopen(path, "rb");
  if (!fp) {
    LOG(WARNING) << "File: " << path << " not exists.";
    return 0;
  }

  int freq = 0;
  int items_read = fscanf(fp, "%d", &freq);
  if (items_read != 1) {
    LOG(WARNING) << "Read file: " << path << " failed.";
  }
  fclose(fp);
  return freq;
}

MaceStatus SetThreadAffinity(cpu_set_t mask) {
#if defined(__ANDROID__)
  pid_t pid = gettid();
#else
  pid_t pid = syscall(SYS_gettid);
#endif
  int err = sched_setaffinity(pid, sizeof(mask), &mask);
  if (err) {
    LOG(WARNING) << "set affinity error: " << strerror(errno);
    return MACE_INVALID_ARGS;
  } else {
    return MACE_SUCCESS;
  }
}

}  // namespace

MaceStatus GetCPUBigLittleCoreIDs(std::vector<int> *big_core_ids,
                                  std::vector<int> *little_core_ids) {
  MACE_CHECK_NOTNULL(big_core_ids);
  MACE_CHECK_NOTNULL(little_core_ids);
  int cpu_count = GetCPUCount();
  std::vector<int> cpu_max_freq(cpu_count);

  // set cpu max frequency
  for (int i = 0; i < cpu_count; ++i) {
    cpu_max_freq[i] = GetCPUMaxFreq(i);
    if (cpu_max_freq[i] == 0) {
      LOG(WARNING) << "Cannot get CPU" << i
                   << "'s max frequency info, maybe it is offline.";
      return MACE_INVALID_ARGS;
    }
  }

  int big_core_freq =
      *(std::max_element(cpu_max_freq.begin(), cpu_max_freq.end()));
  int little_core_freq =
      *(std::min_element(cpu_max_freq.begin(), cpu_max_freq.end()));

  big_core_ids->reserve(cpu_count);
  little_core_ids->reserve(cpu_count);
  for (int i = 0; i < cpu_count; ++i) {
    if (cpu_max_freq[i] == little_core_freq) {
      little_core_ids->push_back(i);
    }
    if (cpu_max_freq[i] == big_core_freq) {
      big_core_ids->push_back(i);
    }
  }

  return MACE_SUCCESS;
}

MaceStatus SetOpenMPThreadsAndAffinityCPUs(int omp_num_threads,
                                           const std::vector<int> &cpu_ids) {
#ifdef MACE_ENABLE_OPENMP
  VLOG(1) << "Set OpenMP threads number: " << omp_num_threads
          << ", CPU core IDs: " << MakeString(cpu_ids);
  omp_set_num_threads(omp_num_threads);
#else
  MACE_UNUSED(omp_num_threads);
  LOG(WARNING) << "Set OpenMP threads number failed: OpenMP not enabled.";
#endif

  // compute mask
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (auto cpu_id : cpu_ids) {
    CPU_SET(cpu_id, &mask);
  }
#ifdef MACE_ENABLE_OPENMP
  std::vector<MaceStatus> status(omp_num_threads);
#pragma omp parallel for
  for (int i = 0; i < omp_num_threads; ++i) {
    VLOG(1) << "Set affinity for OpenMP thread " << omp_get_thread_num()
            << "/" << omp_get_num_threads();
    status[i] = SetThreadAffinity(mask);
  }
  for (int i = 0; i < omp_num_threads; ++i) {
    if (status[i] != MACE_SUCCESS)
      return MACE_INVALID_ARGS;
  }
  return MACE_SUCCESS;
#else
  MaceStatus status = SetThreadAffinity(mask);
  VLOG(1) << "Set affinity without OpenMP: " << mask.__bits[0];
  return status;
#endif
}

MaceStatus SetOpenMPThreadsAndAffinityPolicy(int omp_num_threads_hint,
                                             CPUAffinityPolicy policy) {
  if (policy == CPUAffinityPolicy::AFFINITY_NONE) {
#ifdef MACE_ENABLE_OPENMP
    if (omp_num_threads_hint > 0) {
      omp_set_num_threads(std::min(omp_num_threads_hint, omp_get_num_procs()));
    }
#else
    LOG(WARNING) << "Set OpenMP threads number failed: OpenMP not enabled.";
#endif
    return MACE_SUCCESS;
  }

  std::vector<int> big_core_ids;
  std::vector<int> little_core_ids;
  MaceStatus res = GetCPUBigLittleCoreIDs(&big_core_ids, &little_core_ids);
  if (res != MACE_SUCCESS) {
    return res;
  }

  std::vector<int> use_cpu_ids;
  if (policy == CPUAffinityPolicy::AFFINITY_BIG_ONLY) {
    use_cpu_ids = std::move(big_core_ids);
  } else {
    use_cpu_ids = std::move(little_core_ids);
  }

  if (omp_num_threads_hint <= 0 ||
      omp_num_threads_hint > static_cast<int>(use_cpu_ids.size())) {
    omp_num_threads_hint = use_cpu_ids.size();
  }

  return SetOpenMPThreadsAndAffinityCPUs(omp_num_threads_hint, use_cpu_ids);
}

}  // namespace mace

