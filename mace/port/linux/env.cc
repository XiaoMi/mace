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

#include "mace/port/linux/env.h"

#include <execinfo.h>
#include <sys/time.h>

#include <cstddef>
#include <string>
#include <vector>

#include "mace/port/env.h"
#include "mace/port/posix/backtrace.h"
#include "mace/port/posix/file_system.h"
#include "mace/port/posix/time.h"
#include "mace/utils/macros.h"

namespace mace {
namespace port {

// In our embedded linux device, SchedSetAffinity has side effects
// on performance, so we override this method to do nothing. You
// can try to comment this function, perhaps you could get a better
// performance as we do in Android devices.
MaceStatus LinuxEnv::SchedSetAffinity(const std::vector<size_t> &cpu_ids) {
  MACE_UNUSED(cpu_ids);

  return MaceStatus::MACE_SUCCESS;
}

LogWriter *LinuxEnv::GetLogWriter() {
  return &log_writer_;
}

std::vector<std::string> LinuxEnv::GetBackTraceUnsafe(int max_steps) {
  return mace::port::posix::GetBackTraceUnsafe(max_steps);
}

Env *Env::Default() {
  static LinuxEnv linux_env;
  return &linux_env;
}

}  // namespace port
}  // namespace mace
