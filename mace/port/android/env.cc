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

#include "mace/port/android/env.h"

#include <errno.h>
#include <unwind.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <sys/types.h>

#ifdef __hexagon__
#include <HAP_perf.h>
#else
#include <sys/time.h>
#endif

#include <cstdint>
#include <memory>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

#include "mace/port/android/malloc_logger.h"
#include "mace/port/posix/time.h"
#include "mace/utils/macros.h"
#include "mace/utils/memory.h"
#include "mace/utils/logging.h"

namespace mace {
namespace port {

LogWriter *AndroidEnv::GetLogWriter() {
  return &log_writer_;
}

namespace {

struct BacktraceState {
  void **current;
  void **end;
};

_Unwind_Reason_Code UnwindCallback(struct _Unwind_Context *context, void *arg) {
  BacktraceState *state = static_cast<BacktraceState *>(arg);
  uintptr_t pc = _Unwind_GetIP(context);
  if (pc) {
    if (state->current == state->end) {
      return _URC_END_OF_STACK;
    } else {
      *state->current++ = reinterpret_cast<void *>(pc);
    }
  }
  return _URC_NO_REASON;
}

size_t BackTrace(void **buffer, size_t max) {
  BacktraceState state = {buffer, buffer + max};
  _Unwind_Backtrace(UnwindCallback, &state);

  return state.current - buffer;
}

bool CpuIsolate(size_t cpu_id) {
  std::string cpuinfo_isolate_conf = MakeString(
      "/sys/devices/system/cpu/cpu",
      cpu_id,
      "/isolate");
  std::ifstream isolate_file(cpuinfo_isolate_conf);
  int isolate_switch = 0;
  if (isolate_file.is_open()) {
    std::string line;
    if (std::getline(isolate_file, line)) {
      isolate_switch = strtol(line.c_str(), nullptr, 0);
    }
    isolate_file.close();
  }

  return (isolate_switch != 0);
}

}  // namespace

MaceStatus AndroidEnv::GetCPUMaxFreq(std::vector<float> *max_freqs) {
  MACE_RETURN_IF_ERROR(LinuxBaseEnv::GetCPUMaxFreq(max_freqs));

  size_t cpu_num = (max_freqs != nullptr) ? max_freqs->size() : 0;
  if (cpu_num > 0) {
    for (size_t i = 0; i < cpu_num; ++i) {
      if (CpuIsolate(i)) {
        (*max_freqs)[i] = 0;
      }
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

std::vector<std::string> AndroidEnv::GetBackTraceUnsafe(int max_steps) {
  std::vector<void *> buffer(max_steps, 0);
  int steps = BackTrace(buffer.data(), max_steps);

  std::vector<std::string> bt;
  for (int i = 0; i < steps; ++i) {
    std::ostringstream os;

    const void *addr = buffer[i];
    const char *symbol = "";
    Dl_info info;
    if (dladdr(addr, &info) && info.dli_sname) {
      symbol = info.dli_sname;
    }

    os << "pc " << addr << " " << symbol;

    bt.push_back(os.str());
  }

  return bt;
}

std::unique_ptr<MallocLogger> AndroidEnv::NewMallocLogger(
    std::ostringstream *oss,
    const std::string &name) {
  return make_unique<AndroidMallocLogger>(oss, name);
}

Env *Env::Default() {
  static AndroidEnv android_env;
  return &android_env;
}

}  // namespace port
}  // namespace mace
