//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <omp.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

#include "mace/core/runtime/cpu/cpu_runtime.h"
#include "mace/public/mace.h"
#include "mace/utils/logging.h"
namespace mace {

namespace {

int GetCPUMaxFreq(int cpu_id) {
  char path[64];
  snprintf(path, sizeof(path),
          "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq",
          cpu_id);
  FILE *fp = fopen(path, "rb");
  MACE_CHECK(fp, "File: ", path, " not exists");

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
  int sys_call_res;
  pid_t pid = gettid();
  int err = sched_setaffinity(pid, sizeof(mask), &mask);
  MACE_CHECK(err == 0, "set affinity error: ", errno);
}

}  // namespace

void SetOmpThreadsAndAffinity(int omp_num_threads,
                              CPUPowerOption power_option) {
  int cpu_count = omp_get_num_procs();
  std::vector<int> sorted_cpu_ids;
  sorted_cpu_ids.resize(cpu_count);
  int big_core_offset;
  SortCPUIdsByMaxFreqAsc(&sorted_cpu_ids, &big_core_offset);

  std::vector<int> use_cpu_ids;
  if (power_option == CPUPowerOption::DEFAULT) {
    use_cpu_ids = sorted_cpu_ids;
  } else if (power_option == CPUPowerOption::HIGH_PERFORMANCE) {
    use_cpu_ids = std::vector<int>(sorted_cpu_ids.begin() + big_core_offset,
                                   sorted_cpu_ids.end());
  } else {
    if (big_core_offset > 0) {
      use_cpu_ids = std::vector<int>(sorted_cpu_ids.begin(),
                                     sorted_cpu_ids.begin() + big_core_offset);
    } else {
      use_cpu_ids = sorted_cpu_ids;
    }
  }

  if (omp_num_threads > use_cpu_ids.size()) {
    LOG(WARNING) << "set omp num threads greater than num of cpus can use: "
                 << use_cpu_ids.size();
  }
  omp_set_num_threads(omp_num_threads);

  // compute mask
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (auto cpu_id : use_cpu_ids) {
    CPU_SET(cpu_id, &mask);
  }
  VLOG(3) << "Set cpu affinity with mask: " << mask.__bits[0];

#pragma omp parallel for
  for (int i = 0; i < omp_num_threads; ++i) {
    SetThreadAffinity(mask);
  }
}

}  // namespace mace

