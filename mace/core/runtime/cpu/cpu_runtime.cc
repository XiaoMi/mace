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

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <algorithm>
#include <utility>
#include <vector>

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
        LOG(ERROR) << "Access " << path << " failed, errno: " << errno;
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
  fscanf(fp, "%d", &freq);
  fclose(fp);
  return freq;
}

void SortCPUIdsByMaxFreqAsc(std::vector<int> *cpu_ids, int *big_core_offset) {
  MACE_CHECK_NOTNULL(cpu_ids);
  int cpu_count = cpu_ids->size();
  std::vector<int> cpu_max_freq;
  cpu_max_freq.resize(cpu_count);

  // set cpu max frequency
  for (int i = 0; i < cpu_count; ++i) {
    cpu_max_freq[i] = GetCPUMaxFreq(i);
    (*cpu_ids)[i] = i;
  }

  // sort cpu ids by max frequency asc, bubble sort
  for (int i = 0; i < cpu_count - 1; ++i) {
    for (int j = i + 1; j < cpu_count; ++j) {
      if (cpu_max_freq[i] > cpu_max_freq[j]) {
        int tmp = (*cpu_ids)[i];
        (*cpu_ids)[i] = (*cpu_ids)[j];
        (*cpu_ids)[j] = tmp;

        tmp = cpu_max_freq[i];
        cpu_max_freq[i] = cpu_max_freq[j];
        cpu_max_freq[j] = tmp;
      }
    }
  }

  *big_core_offset = 0;
  for (int i = 1; i < cpu_count; ++i) {
    if (cpu_max_freq[i] > cpu_max_freq[i - 1]) {
      *big_core_offset = i;
      break;
    }
  }
}

void SetThreadAffinity(cpu_set_t mask) {
#if defined(__ANDROID__)
  pid_t pid = gettid();
#else
  pid_t pid = syscall(SYS_gettid);
#endif
  int err = sched_setaffinity(pid, sizeof(mask), &mask);
  MACE_CHECK(err == 0, "set affinity error: ", errno);
}

}  // namespace

MaceStatus GetCPUBigLittleCoreIDs(std::vector<int> *big_core_ids,
                                  std::vector<int> *little_core_ids) {
  MACE_CHECK_NOTNULL(big_core_ids);
  MACE_CHECK_NOTNULL(little_core_ids);
#ifdef MACE_ENABLE_OPENMP
  int cpu_count = omp_get_num_procs();
#else
  int cpu_count = GetCPUCount();
#endif
  std::vector<int> cpu_max_freq(cpu_count);
  std::vector<int> cpu_ids(cpu_count);

  // set cpu max frequency
  for (int i = 0; i < cpu_count; ++i) {
    cpu_max_freq[i] = GetCPUMaxFreq(i);
    if (cpu_max_freq[i] == 0) {
      LOG(WARNING) << "Cannot get cpu" << i
                   << "'s max frequency info, maybe it is offline.";
      return MACE_INVALID_ARGS;
    }
    cpu_ids[i] = i;
  }

  // sort cpu ids by max frequency asc
  std::sort(cpu_ids.begin(), cpu_ids.end(),
            [&cpu_max_freq](int a, int b) {
              return cpu_max_freq[a] < cpu_max_freq[b];
            });

  big_core_ids->reserve(cpu_count);
  little_core_ids->reserve(cpu_count);
  int little_core_freq = cpu_max_freq.front();
  int big_core_freq = cpu_max_freq.back();
  for (int i = 0; i < cpu_count; ++i) {
    if (cpu_max_freq[i] == little_core_freq) {
      little_core_ids->push_back(cpu_ids[i]);
    }
    if (cpu_max_freq[i] == big_core_freq) {
      big_core_ids->push_back(cpu_ids[i]);
    }
  }

  return MACE_SUCCESS;
}

void SetOpenMPThreadsAndAffinityCPUs(int omp_num_threads,
                                     const std::vector<int> &cpu_ids) {
  VLOG(1) << "Set OpenMP threads number: " << omp_num_threads
          << ", CPU core IDs: " << MakeString(cpu_ids);

#ifdef MACE_ENABLE_OPENMP
  omp_set_num_threads(omp_num_threads);
#else
  LOG(WARNING) << "OpenMP not enabled. Set OpenMP threads number failed.";
#endif

  // compute mask
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (auto cpu_id : cpu_ids) {
    CPU_SET(cpu_id, &mask);
  }

#ifdef MACE_ENABLE_OPENMP
#pragma omp parallel for
  for (int i = 0; i < omp_num_threads; ++i) {
    SetThreadAffinity(mask);
  }
#else
  SetThreadAffinity(mask);
  LOG(INFO) << "SetThreadAffinity: " << mask.__bits[0];
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
    LOG(WARNING) << "OpenMP not enabled. Set OpenMP threads number failed.";
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
  SetOpenMPThreadsAndAffinityCPUs(omp_num_threads_hint, use_cpu_ids);
  return MACE_SUCCESS;
}

MaceStatus SetOpenMPThreadPolicy(int num_threads_hint,
                                 CPUAffinityPolicy policy) {
  VLOG(1) << "Set CPU openmp num_threads_hint: " << num_threads_hint
          << ", affinity policy: " << policy;
  return SetOpenMPThreadsAndAffinityPolicy(num_threads_hint, policy);
}

void SetOpenMPThreadAffinity(int num_threads, const std::vector<int> &cpu_ids) {
  return SetOpenMPThreadsAndAffinityCPUs(num_threads, cpu_ids);
}

MaceStatus GetBigLittleCoreIDs(std::vector<int> *big_core_ids,
                               std::vector<int> *little_core_ids) {
  return GetCPUBigLittleCoreIDs(big_core_ids, little_core_ids);
}

}  // namespace mace

