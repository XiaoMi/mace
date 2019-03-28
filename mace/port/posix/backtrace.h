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

#ifndef MACE_PORT_POSIX_BACKTRACE_H_
#define MACE_PORT_POSIX_BACKTRACE_H_

#include <execinfo.h>

#include <string>
#include <vector>

namespace mace {
namespace port {
namespace posix {

inline std::vector<std::string> GetBackTraceUnsafe(int max_steps) {
  std::vector<void *> buffer(max_steps, 0);
  int steps = backtrace(buffer.data(), max_steps);

  std::vector<std::string> bt;
  char **symbols = backtrace_symbols(buffer.data(), steps);
  if (symbols != nullptr) {
    for (int i = 0; i < steps; i++) {
      bt.push_back(symbols[i]);
    }
  }
  return bt;
}

}  // namespace posix
}  // namespace port
}  // namespace mace

#endif  // MACE_PORT_POSIX_BACKTRACE_H_
