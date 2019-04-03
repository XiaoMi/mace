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

#ifndef MACE_PORT_DARWIN_ENV_H_
#define MACE_PORT_DARWIN_ENV_H_

#include <string>
#include <vector>

#include "mace/port/env.h"
#include "mace/port/logger.h"
#include "mace/port/posix/file_system.h"

namespace mace {
namespace port {

class DarwinEnv : public Env {
 public:
  int64_t NowMicros() override;
  MaceStatus GetCPUMaxFreq(std::vector<float> *max_freqs) override;
  FileSystem *GetFileSystem() override;
  LogWriter *GetLogWriter() override;
  std::vector<std::string> GetBackTraceUnsafe(int max_steps) override;

 private:
  PosixFileSystem posix_file_system_;
  LogWriter log_writer_;
};

}  // namespace port
}  // namespace mace

#endif  // MACE_PORT_DARWIN_ENV_H_
