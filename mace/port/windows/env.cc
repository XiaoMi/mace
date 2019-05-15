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

#include "mace/port/windows/env.h"

#include <windows.h>

#include <ctime>
#include <string>
#include <vector>

namespace mace {
namespace port {

int64_t WindowsEnv::NowMicros() {
  int64_t sec = time(nullptr);
  SYSTEMTIME sys_time;
  GetLocalTime(&sys_time);
  int64_t msec = sys_time.wMilliseconds;
  return sec * 1000000 + msec * 1000;
}

FileSystem *WindowsEnv::GetFileSystem() {
  return &windows_file_system_;
}

LogWriter *WindowsEnv::GetLogWriter() {
  return &log_writer_;
}

std::vector<std::string> WindowsEnv::GetBackTraceUnsafe(int max_steps) {
  std::vector<std::string> empty;
  return empty;
}

Env *Env::Default() {
  static WindowsEnv windows_env;
  return &windows_env;
}

}  // namespace port
}  // namespace mace
