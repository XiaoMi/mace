// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include "mace/public/mace.h"
#include "mace/utils/env_time.h"
#include "mace/utils/string_util.h"
#include "mace/utils/utils.h"

#undef ERROR

namespace mace {

const int INFO = 0;            // base_logging::INFO;
const int WARNING = 1;         // base_logging::WARNING;
const int ERROR = 2;           // base_logging::ERROR;
const int FATAL = 3;           // base_logging::FATAL;
const int NUM_SEVERITIES = 4;  // base_logging::NUM_SEVERITIES;

namespace logging {

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char *fname, int line, int severity);
  ~LogMessage();

  // Returns the minimum log level for VLOG statements.
  // E.g., if MinVLogLevel() is 2, then VLOG(2) statements will produce output,
  // but VLOG(3) will not. Defaults to 0.
  static int64_t MinVLogLevel();

 protected:
  void GenerateLogMessage();

 private:
  const char *fname_;
  int line_;
  int severity_;
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char *file, int line);
  ~LogMessageFatal();
};

#define _MACE_LOG_INFO \
  ::mace::logging::LogMessage(__FILE__, __LINE__, mace::INFO)
#define _MACE_LOG_WARNING \
  ::mace::logging::LogMessage(__FILE__, __LINE__, mace::WARNING)
#define _MACE_LOG_ERROR \
  ::mace::logging::LogMessage(__FILE__, __LINE__, mace::ERROR)
#define _MACE_LOG_FATAL ::mace::logging::LogMessageFatal(__FILE__, __LINE__)

#define _MACE_LOG_QFATAL _MACE_LOG_FATAL

#define LOG(severity) _MACE_LOG_##severity

// Set MACE_CPP_MIN_VLOG_LEVEL environment to update minimum log
// level
// of VLOG
#define VLOG_IS_ON(lvl) ((lvl) <= ::mace::logging::LogMessage::MinVLogLevel())

#define VLOG(lvl)      \
  if (VLOG_IS_ON(lvl)) \
  ::mace::logging::LogMessage(__FILE__, __LINE__, mace::INFO)

// MACE_CHECK/MACE_ASSERT dies with a fatal error if condition is not true.
// MACE_ASSERT is controlled by NDEBUG ('-c opt' for bazel) while MACE_CHECK
// will be executed regardless of compilation mode.
// Therefore, it is safe to do things like:
//    MACE_CHECK(fp->Write(x) == 4)
//    MACE_CHECK(fp->Write(x) == 4, "Write failed")
// which are not correct for MACE_ASSERT.
#define MACE_CHECK(condition, ...) \
  if (!(condition))                \
  LOG(FATAL) << "Check failed: " #condition " " << MakeString(__VA_ARGS__)

#ifndef NDEBUG
#define MACE_ASSERT(condition, ...) \
  if (!(condition))                 \
  LOG(FATAL) << "Assert failed: " #condition " " << MakeString(__VA_ARGS__)
#else
#define MACE_ASSERT(condition, ...) ((void)0)
#endif

template <typename T>
T &&CheckNotNull(const char *file, int line, const char *exprtext, T &&t) {
  if (t == nullptr) {
    LogMessageFatal(file, line) << std::string(exprtext);
  }
  return std::forward<T>(t);
}

#define MACE_CHECK_NOTNULL(val)                     \
  ::mace::logging::CheckNotNull(__FILE__, __LINE__, \
                                "'" #val "' Must be non NULL", (val))

#define MACE_NOT_IMPLEMENTED MACE_CHECK(false, "not implemented")

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

  DISABLE_COPY_AND_ASSIGN(LatencyLogger);
};

#define MACE_LATENCY_LOGGER(vlog_level, ...)              \
  mace::logging::LatencyLogger latency_logger_##__line__( \
      vlog_level, VLOG_IS_ON(vlog_level) ? mace::MakeString(__VA_ARGS__) : "")

}  // namespace logging
}  // namespace mace

#endif  // MACE_UTILS_LOGGING_H_
