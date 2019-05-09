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
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#endif

#include <sys/stat.h>

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
  virtual MaceStatus AdviseFree(void *addr, size_t length);
  virtual MaceStatus GetCPUMaxFreq(std::vector<float> *max_freqs);
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

inline MaceStatus AdviseFree(void *addr, size_t length) {
  return port::Env::Default()->AdviseFree(addr, length);
}

inline MaceStatus GetCPUMaxFreq(std::vector<float> *max_freqs) {
  return port::Env::Default()->GetCPUMaxFreq(max_freqs);
}

inline MaceStatus SchedSetAffinity(const std::vector<size_t> &cpu_ids) {
  return port::Env::Default()->SchedSetAffinity(cpu_ids);
}

inline port::FileSystem *GetFileSystem() {
  return port::Env::Default()->GetFileSystem();
}

inline MaceStatus Memalign(void **memptr, size_t alignment, size_t size) {
#ifdef _WIN32
  *memptr = _aligned_malloc(size, alignment);
  if (*memptr == nullptr) {
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  } else {
    return MaceStatus::MACE_SUCCESS;
  }
#else
#if defined(__ANDROID__) || defined(__hexagon__)
  *memptr = memalign(alignment, size);
  if (*memptr == nullptr) {
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  } else {
    return MaceStatus::MACE_SUCCESS;
  }
#else
  int error = posix_memalign(memptr, alignment, size);
  if (error != 0) {
    if (*memptr != nullptr) {
      free(*memptr);
      *memptr = nullptr;
    }
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  } else {
    return MaceStatus::MACE_SUCCESS;
  }
#endif
#endif
}

inline MaceStatus GetEnv(const char *name, std::string *value) {
#ifdef _WIN32
  char *val;
  size_t len;
  errno_t error = _dupenv_s(&val, &len, name);
  if (error != 0) {
    return MaceStatus::MACE_RUNTIME_ERROR;
  } else {
    if (val != nullptr) {
      *value = std::string(val);
      free(val);
    }
    return MaceStatus::MACE_SUCCESS;
  }
#else
  char *val = getenv(name);
  if (val != nullptr) {
    *value = std::string(val);
  }
  return MaceStatus::MACE_SUCCESS;
#endif
}

#if defined(_WIN32) && !defined(S_ISREG)
#define S_ISREG(m) (((m) & 0170000) == (0100000))
#endif
}  // namespace mace

#endif  // MACE_PORT_ENV_H_
