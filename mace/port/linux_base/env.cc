// Copyright 2019 The MACE Authors. All Rights Reserved.
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

#include "mace/port/linux_base/env.h"

#include <sys/time.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "mace/port/posix/file_system.h"
#include "mace/port/posix/time.h"
#include "mace/utils/logging.h"

namespace mace {
namespace port {


namespace {

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
  VLOG(1) << "CPU cores: " << cpu_count;
  return cpu_count;
}

}  // namespace

int64_t LinuxBaseEnv::NowMicros() {
  return mace::port::posix::NowMicros();
}

FileSystem *LinuxBaseEnv::GetFileSystem() {
  return &posix_file_system_;
}

MaceStatus LinuxBaseEnv::GetCPUMaxFreq(std::vector<float> *max_freqs) {
  MACE_CHECK_NOTNULL(max_freqs);
  int cpu_count = GetCPUCount();
  if (cpu_count < 0) {
    return MaceStatus::MACE_RUNTIME_ERROR;
  }
  for (int cpu_id = 0; cpu_id < cpu_count; ++cpu_id) {
    std::string cpuinfo_max_freq_sys_conf = MakeString(
        "/sys/devices/system/cpu/cpu",
        cpu_id,
        "/cpufreq/cpuinfo_max_freq");
    std::ifstream f(cpuinfo_max_freq_sys_conf);
    if (!f.is_open()) {
      LOG(ERROR) << "failed to open " << cpuinfo_max_freq_sys_conf;
      return MaceStatus::MACE_RUNTIME_ERROR;
    }
    std::string line;
    if (std::getline(f, line)) {
      float freq = strtof(line.c_str(), nullptr);
      max_freqs->push_back(freq);
    }
    if (f.bad()) {
      LOG(ERROR) << "failed to read " << cpuinfo_max_freq_sys_conf;
    }
    f.close();
  }

  VLOG(1) << "CPU freq: " << MakeString(*max_freqs);

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace port
}  // namespace mace
