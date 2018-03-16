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
  if (!fp) return 0;

  int freq = 0;
  fscanf(fp, "%d", &freq);
  fclose(fp);
  return freq;
}

void SortCPUIdsByMaxFreqAsc(std::vector<int> *cpu_ids) {
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
}

void SetThreadAffinity(cpu_set_t mask) {
  int sys_call_res;
  pid_t pid = gettid();

  // TODO(chenghui): when set omp num threads to 1,
  // sometiomes return EINVAL(22) error.
  // https://linux.die.net/man/2/sched_setaffinity
  sys_call_res = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
  if (sys_call_res != 0) {
    LOG(FATAL) << "syscall setaffinity error: " << sys_call_res << ' ' << errno;
  }
}

}  // namespace

void SetCPURuntime(int omp_num_threads, CPUPowerOption power_option) {
  int cpu_count = omp_get_num_procs();
  LOG(INFO) << "cpu_count: " << cpu_count;
  std::vector<int> sorted_cpu_ids;
  sorted_cpu_ids.resize(cpu_count);
  SortCPUIdsByMaxFreqAsc(&sorted_cpu_ids);

  std::vector<int> use_cpu_ids;
  if (power_option == CPUPowerOption::DEFAULT || omp_num_threads >= cpu_count) {
    use_cpu_ids = sorted_cpu_ids;
    omp_num_threads = cpu_count;
  } else if (power_option == CPUPowerOption::HIGH_PERFORMANCE) {
    use_cpu_ids =
        std::vector<int>(sorted_cpu_ids.begin() + cpu_count - omp_num_threads,
                         sorted_cpu_ids.end());
  } else {
    use_cpu_ids = std::vector<int>(sorted_cpu_ids.begin(),
                                   sorted_cpu_ids.begin() + omp_num_threads);
  }

  omp_set_num_threads(omp_num_threads);
  // compute mask
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (auto cpu_id : use_cpu_ids) {
    CPU_SET(cpu_id, &mask);
  }
  LOG(INFO) << "use cpus mask: " << mask.__bits[0];

#pragma omp parallel for
  for (int i = 0; i < omp_num_threads; ++i) {
    SetThreadAffinity(mask);
  }
}

}  // namespace mace

