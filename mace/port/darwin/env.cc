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

#include "mace/port/darwin/env.h"

#include <execinfo.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <stdint.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <sys/types.h>

#include <cstddef>
#include <string>
#include <vector>

#include "mace/port/posix/backtrace.h"
#include "mace/port/posix/file_system.h"
#include "mace/port/posix/time.h"
#include "mace/utils/logging.h"

namespace mace {
namespace port {

namespace {

constexpr const char kCpuFrequencyMax[] = "hw.cpufrequency_max";
constexpr const char kCpuActiveNum[] = "hw.activecpu";

}

int64_t DarwinEnv::NowMicros() {
  return mace::port::posix::NowMicros();
}

// we can't get the frequancy of every cpu on darwin, so this method
// return a fake frequancy data.
MaceStatus DarwinEnv::GetCPUMaxFreq(std::vector<float> *cpu_infos) {
  MACE_CHECK_NOTNULL(cpu_infos);

  float freq = 0;
  size_t size = sizeof(freq);
  int ret = sysctlbyname(kCpuFrequencyMax, &freq, &size, NULL, 0);
  if (ret < 0) {
    LOG(ERROR) << "failed to get property: " << kCpuFrequencyMax;
    return MaceStatus::MACE_RUNTIME_ERROR;
  }

  uint64_t cpu_num = 0;
  size = sizeof(cpu_num);
  ret = sysctlbyname(kCpuActiveNum, &cpu_num, &size, NULL, 0);
  if (ret < 0) {
    LOG(ERROR) << "failed to get property: " << kCpuActiveNum;
    return MaceStatus::MACE_RUNTIME_ERROR;
  }

  for (int i = 0; i < cpu_num; ++i) {
    cpu_infos->push_back(freq);
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus DarwinEnv::SchedSetAffinity(
    const std::vector<size_t> &cpu_ids) {
  unsigned int tag = 0;
  for (size_t i = 0; i < cpu_ids.size(); ++i) {
    tag += (cpu_ids[i] << i);
  }

#ifdef MACE_OS_MAC
  pthread_t thread = pthread_self();
  mach_port_t mach_port = pthread_mach_thread_np(thread);
  thread_affinity_policy_data_t policy_data = {(integer_t) tag};
  int ret = thread_policy_set(mach_port,
                              THREAD_AFFINITY_POLICY,
                              (thread_policy_t) & policy_data,
                              1);
  if (ret) {
    LOG(INFO) << "thread_policy_set failed: " << strerror(errno);
    return MaceStatus::MACE_RUNTIME_ERROR;
  }
#endif

  return MaceStatus::MACE_SUCCESS;
}

FileSystem *DarwinEnv::GetFileSystem() {
  return &posix_file_system_;
}

LogWriter *DarwinEnv::GetLogWriter() {
  return &log_writer_;
}

std::vector<std::string> DarwinEnv::GetBackTraceUnsafe(int max_steps) {
  return mace::port::posix::GetBackTraceUnsafe(max_steps);
}

Env *Env::Default() {
  static DarwinEnv darwin_env;
  return &darwin_env;
}

}  // namespace port
}  // namespace mace
