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

#ifndef MACE_UTILS_LOGGING_H_
#define MACE_UTILS_LOGGING_H_

#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include "mace/port/env.h"
#include "mace/port/logger.h"
#include "mace/utils/macros.h"
#include "mace/utils/memory.h"
#include "mace/utils/string_util.h"


namespace mace {
namespace logging_internal {

#define LOG(severity) \
  ::mace::port::Logger(__FILE__, __LINE__, mace::severity)

#define LOG_PTR(severity) \
  make_unique<mace::port::Logger>(__FILE__, __LINE__, mace::severity)

#define VLOG_IS_ON(vll) (mace::ShouldGenerateVLogMessage(vll))
#define VLOG(vll) if (VLOG_IS_ON(vll)) LOG(INFO)

// MACE_CHECK/MACE_ASSERT dies with a fatal error if condition is not true.
// MACE_ASSERT is controlled by NDEBUG ('-c opt' for bazel) while MACE_CHECK
// will be executed regardless of compilation mode.
// Therefore, it is safe to do things like:
//    MACE_CHECK(fp->Write(x) == 4)
//    MACE_CHECK(fp->Write(x) == 4, "Write failed")
// which are not safe for MACE_ASSERT.
#define MACE_CHECK(condition, ...) \
  if (!(condition)) \
  LOG(FATAL) << "Check failed: " #condition " " << mace::MakeString(__VA_ARGS__)

#ifndef NDEBUG
#define MACE_ASSERT(condition, ...) \
  if (!(condition)) \
  LOG(FATAL) << "Assert failed: " #condition " " \
             << mace::MakeString(__VA_ARGS__)
#else
#define MACE_ASSERT(condition, ...) ((void)0)
#endif

template <typename T>
T &&CheckNotNull(const char *file, int line, const char *exprtext, T &&t) {
  if (t == nullptr) {
    ::mace::port::Logger(file, line, FATAL) << std::string(exprtext);
  }
  return std::forward<T>(t);
}

#define MACE_CHECK_NOTNULL(val) \
  ::mace::logging_internal::CheckNotNull(__FILE__, __LINE__, \
                                         "'" #val "' Must not be NULL", (val))

#define MACE_NOT_IMPLEMENTED MACE_CHECK(false, "not implemented")

#define MACE_CHECK_SUCCESS(stmt)                             \
  {                                                          \
    MaceStatus status = (stmt);                              \
    if (status != MaceStatus::MACE_SUCCESS) {                \
      LOG(FATAL) << #stmt << " failed with error: "          \
              << status.information();                       \
    }                                                        \
  }

#define MACE_RETURN_IF_ERROR(stmt)                           \
  {                                                          \
    MaceStatus status = (stmt);                              \
    if (status != MaceStatus::MACE_SUCCESS) {                \
      VLOG(0) << #stmt << " failed with error: "             \
              << status.information();                       \
      return status;                                         \
    }                                                        \
  }

class LatencyLogger {
 public:
  LatencyLogger(int vlog_level, const std::string &message)
      : vlog_level_(vlog_level), message_(message) {
    if (VLOG_IS_ON(vlog_level_)) {
      start_micros_ = NowMicros();
      VLOG(vlog_level_) << message_ << " started";
    }
  }
  ~LatencyLogger() {
    if (VLOG_IS_ON(vlog_level_)) {
      int64_t stop_micros = NowMicros();
      VLOG(vlog_level_) << message_
                        << " latency: " << stop_micros - start_micros_ << " us";
    }
  }

 private:
  const int vlog_level_;
  const std::string message_;
  int64_t start_micros_;

  MACE_DISABLE_COPY_AND_ASSIGN(LatencyLogger);
};

#define MACE_LATENCY_LOGGER(vlog_level, ...)                                  \
  mace::logging_internal::LatencyLogger latency_logger_##__line__(            \
      vlog_level, VLOG_IS_ON(vlog_level) ? mace::MakeString(__VA_ARGS__) : "")


#ifdef MACE_ENABLE_MALLOC_LOGGING
#define MACE_MEMORY_LOGGING_GUARD()                                      \
  auto malloc_logger_##__line__ = port::Env::Default()->NewMallocLogger( \
      ::mace::port::Logger(__FILE__, __LINE__, mace::INFO), \
      std::string(__FILE__) + ":" + std::string(__func__));
#else
#define MACE_MEMORY_LOGGING_GUARD()
#endif

}  // namespace logging_internal
}  // namespace mace

#endif  // MACE_UTILS_LOGGING_H_
