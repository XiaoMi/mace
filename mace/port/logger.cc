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

#include "mace/port/logger.h"

#include <cstdlib>
#include <iomanip>
#include <string>
#include <vector>

#include "mace/port/env.h"
#include "mace/utils/string_util.h"

namespace mace {
namespace port {

inline bool IsValidLogLevel(const LogLevel level) {
  return level > LogLevel::INVALID_MIN &&
         level < LogLevel::INVALID_MAX;
}

LogLevel LogLevelFromStr(const char *log_level_str) {
  if (log_level_str != nullptr) {
    std::string ls = ToUpper(log_level_str);

    if (ls == "I" || ls == "INFO") {
      return LogLevel::INFO;
    }
    if (ls == "W" || ls == "WARNING") {
      return LogLevel::WARNING;
    }
    if (ls == "E" || ls == "ERROR") {
      return LogLevel::ERROR;
    }
    if (ls == "F" || ls == "FATAL") {
      return LogLevel::FATAL;
    }
  }

  return LogLevel::INVALID_MIN;
}

char LogLevelToShortStr(LogLevel level) {
  if (!IsValidLogLevel(level)) {
    level = LogLevel::INFO;
  }

  return "IWEF"[static_cast<int>(level) - 1];
}

int VLogLevelFromStr(const char *vlog_level_str) {
  if (vlog_level_str != nullptr) {
    return atoi(vlog_level_str);
  }

  return 0;
}


void LogWriter::WriteLogMessage(const char *fname,
                                const int line,
                                const LogLevel severity,
                                const char *message) {
  printf("%c %s:%d] %s\n", LogLevelToShortStr(severity), fname, line, message);
}

Logger::Logger(const char *fname, int line, LogLevel severity)
    : fname_(fname), line_(line), severity_(severity) {}

void Logger::GenerateLogMessage() {
  LogWriter *log_writer = Env::Default()->GetLogWriter();
  log_writer->WriteLogMessage(fname_, line_, severity_, str().c_str());

  // When there is a fatal log, terminate execution
  if (severity_ == LogLevel::FATAL) {
    DealWithFatal();
  }
}

void Logger::DealWithFatal() {
  // When there is a fatal log, log the backtrace and abort.
  LogWriter *log_writer = Env::Default()->GetLogWriter();
  std::vector<std::string> bt = Env::Default()->GetBackTraceUnsafe(50);
  if (!bt.empty()) {
    log_writer->WriteLogMessage(fname_, line_, severity_, "backtrace:");
    for (size_t i = 0; i < bt.size(); ++i) {
      std::ostringstream os;
      os << " " << bt[i];
      log_writer->WriteLogMessage(fname_, line_, severity_, os.str().c_str());
    }
  }

  abort();
}

Logger::~Logger() {
  static const LogLevel min_log_level = MinLogLevelFromEnv();
  if (LogLevelPassThreashold(severity_, min_log_level)) {
    GenerateLogMessage();
  }
}

}  // namespace port
}  // namespace mace
