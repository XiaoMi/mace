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
#include <cerrno>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "mace/core/macros.h"
#include "mace/public/mace.h"
#include "mace/utils/logging.h"

namespace mace {

int MaceOpenMPThreadCount = 1;

struct CPUFreq {
  size_t core_id;
  float freq;
};

namespace {
#if defined(__ANDROID__)
int GetCPUCount() {
  int cpu_count = 0;
  std::string cpu_sys_conf = "/proc/cpuinfo";
  std::ifstream f(cpu_sys_conf);
  if (!f.is_open()) {
    LOG(ERROR) << "failed to open " << cpu_sys_conf;
    return -1;
  }
  std::string line;
  const std::string processor_key = "processor";
  while (std::getline(f, line)) {
    if (line.size() >= processor_key.size()
        && line.compare(0, processor_key.size(), processor_key) == 0) {
      ++cpu_count;
    }
  }
  if (f.bad()) {
    LOG(ERROR) << "failed to read " << cpu_sys_conf;
  }
  if (!f.eof()) {
    LOG(ERROR) << "failed to read end of " << cpu_sys_conf;
  }
  f.close();
  VLOG(2) << "CPU cores: " << cpu_count;
  return cpu_count;
}
#endif

int GetCPUMaxFreq(std::vector<float> *max_freqs) {
#if defined(__ANDROID__)
  int cpu_count = GetCPUCount();
  for (int cpu_id = 0; cpu_id < cpu_count; ++cpu_id) {
    std::string cpuinfo_max_freq_sys_conf = MakeString(
        "/sys/devices/system/cpu/cpu",
        cpu_id,
        "/cpufreq/cpuinfo_max_freq");
    std::ifstream f(cpuinfo_max_freq_sys_conf);
    if (!f.is_open()) {
      LOG(ERROR) << "failed to open " << cpuinfo_max_freq_sys_conf;
      return -1;
    }
    std::string line;
    if (std::getline(f, line)) {
      float freq = atof(line.c_str());
      max_freqs->push_back(freq);
    }
    if (f.bad()) {
      LOG(ERROR) << "failed to read " << cpuinfo_max_freq_sys_conf;
    }
    f.close();
  }
#else
  std::string cpu_sys_conf = "/proc/cpuinfo";
  std::ifstream f(cpu_sys_conf);
  if (!f.is_open()) {
    LOG(ERROR) << "failed to open " << cpu_sys_conf;
    return -1;
  }
  std::string line;
  const std::string freq_key = "cpu MHz";
  while (std::getline(f, line)) {
    if (line.size() >= freq_key.size()
        && line.compare(0, freq_key.size(), freq_key) == 0) {
      size_t pos = line.find(":");
      if (pos != std::string::npos) {
        std::string freq_str = line.substr(pos + 1);
        float freq = atof(freq_str.c_str());
        max_freqs->push_back(freq);
      }
    }
  }
  if (f.bad()) {
    LOG(ERROR) << "failed to read " << cpu_sys_conf;
  }
  if (!f.eof()) {
    LOG(ERROR) << "failed to read end of " << cpu_sys_conf;
  }
  f.close();
#endif

  for (float freq : *max_freqs) {
    VLOG(2) << "CPU freq: " << freq;
  }

  return 0;
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
    return MaceStatus(MaceStatus::MACE_INVALID_ARGS,
                      "set affinity error: " + std::string(strerror(errno)));
  } else {
    return MaceStatus::MACE_SUCCESS;
  }
}

MaceStatus SetOpenMPThreadsAndAffinityCPUs(int omp_num_threads,
                                           const std::vector<size_t> &cpu_ids) {
  MaceOpenMPThreadCount = omp_num_threads;

#ifdef MACE_ENABLE_OPENMP
  VLOG(1) << "Set OpenMP threads number: " << omp_num_threads
          << ", CPU core IDs: " << MakeString(cpu_ids);
  omp_set_schedule(omp_sched_guided, 1);
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
  std::vector<MaceStatus> status(omp_num_threads,
                                 MaceStatus::MACE_INVALID_ARGS);
#pragma omp parallel for
  for (int i = 0; i < omp_num_threads; ++i) {
    VLOG(1) << "Set affinity for OpenMP thread " << omp_get_thread_num()
            << "/" << omp_get_num_threads();
    status[i] = SetThreadAffinity(mask);
  }
  for (int i = 0; i < omp_num_threads; ++i) {
    if (status[i] != MaceStatus::MACE_SUCCESS)
      return MaceStatus::MACE_INVALID_ARGS;
  }
  return MaceStatus::MACE_SUCCESS;
#else
  MaceStatus status = SetThreadAffinity(mask);
  VLOG(1) << "Set affinity without OpenMP: " << mask.__bits[0];
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
  if (GetCPUMaxFreq(&cpu_max_freqs) == -1 || cpu_max_freqs.size() == 0) {
    return MaceStatus::MACE_INVALID_ARGS;
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
    num_threads_hint = cores_to_use;
  } else {
    cores_to_use = num_threads_hint;
  }

  VLOG(2) << "Use " << num_threads_hint << " threads";
  std::vector<size_t> cpu_ids(cores_to_use);
  for (int i = 0; i < cores_to_use; ++i) {
    VLOG(2) << "Bind thread to core: " << cpu_freq[i].core_id << " with freq "
            << cpu_freq[i].freq;
    cpu_ids[i] = cpu_freq[i].core_id;
  }

#ifdef MACE_ENABLE_QUANTIZE
  if (gemm_context) {
    static_cast<gemmlowp::GemmContext*>(gemm_context)->set_max_num_threads(
        num_threads_hint);
  }
#endif  // MACE_ENABLE_QUANTIZE

  return SetOpenMPThreadsAndAffinityCPUs(num_threads_hint, cpu_ids);
}

}  // namespace mace

