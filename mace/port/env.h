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

#ifndef MACE_PORT_ENV_H_
#define MACE_PORT_ENV_H_

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mace/public/mace.h"

namespace mace {
namespace port {

class MallocLogger {
 public:
  MallocLogger() = default;
  virtual ~MallocLogger() = default;
};

class FileSystem;
class LogWriter;

class Env {
 public:
  virtual int64_t NowMicros() = 0;
  virtual MaceStatus GetCpuMaxFreq(std::vector<float> *max_freqs);
  virtual MaceStatus SchedSetAffinity(const std::vector<size_t> &cpu_ids);
  virtual FileSystem *GetFileSystem() = 0;
  virtual LogWriter *GetLogWriter() = 0;
  // Return the current backtrace, will allocate memory inside the call
  // which may fail
  virtual std::vector<std::string> GetBackTraceUnsafe(int max_steps) = 0;
  virtual std::unique_ptr<MallocLogger> NewMallocLogger(
      std::ostringstream *oss,
      const std::string &name);

  static Env *Default();
};

}  // namespace port

inline int64_t NowMicros() {
  return port::Env::Default()->NowMicros();
}

inline MaceStatus GetCpuMaxFreq(std::vector<float> *max_freqs) {
  return port::Env::Default()->GetCpuMaxFreq(max_freqs);
}

inline MaceStatus SchedSetAffinity(const std::vector<size_t> &cpu_ids) {
  return port::Env::Default()->SchedSetAffinity(cpu_ids);
}

inline port::FileSystem *GetFileSystem() {
  return port::Env::Default()->GetFileSystem();
}

}  // namespace mace

#endif  // MACE_PORT_ENV_H_
