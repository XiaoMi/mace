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

#ifndef MACE_PORT_LOGGER_H_
#define MACE_PORT_LOGGER_H_

#include <cstdlib>
#include <cstring>
#include <sstream>

namespace mace {

enum LogLevel {
  INVALID_MIN = 0,
  INFO        = 1,
  WARNING     = 2,
  ERROR       = 3,
  FATAL       = 4,
  INVALID_MAX,
};

namespace port {

inline bool LogLevelPassThreashold(const LogLevel level,
                                   const LogLevel threshold) {
  return level >= threshold;
}

LogLevel LogLevelFromStr(const char *log_level_str);
int VLogLevelFromStr(const char *vlog_level_str);

inline LogLevel MinLogLevelFromEnv() {
  // Read the min log level from env once during the first call to logging.
  static LogLevel log_level = LogLevelFromStr(getenv("MACE_CPP_MIN_LOG_LEVEL"));
  return log_level;
}

inline int MinVLogLevelFromEnv() {
  // Read the min vlog level from env once during the first call to logging.
  static int vlog_level = VLogLevelFromStr(getenv("MACE_CPP_MIN_VLOG_LEVEL"));
  return vlog_level;
}

class LogWriter {
 public:
  LogWriter() = default;
  virtual ~LogWriter() = default;
  virtual void WriteLogMessage(const char *fname,
                               const int line,
                               const LogLevel severity,
                               const char *message);
};

class Logger : public std::ostringstream {
 public:
  Logger(const char *fname, int line, LogLevel severity);
  ~Logger();

 private:
  void GenerateLogMessage();
  void DealWithFatal();

  const char *fname_;
  int line_;
  LogLevel severity_;
};

}  // namespace port

// Whether the log level pass the env configured threshold, can be used for
// short cutting.
inline bool ShouldGenerateLogMessage(LogLevel severity) {
  LogLevel threshold = port::MinLogLevelFromEnv();
  return port::LogLevelPassThreashold(severity, threshold);
}

inline bool ShouldGenerateVLogMessage(int vlog_level) {
  int threshold = port::MinVLogLevelFromEnv();
  return ShouldGenerateLogMessage(INFO) &&
         vlog_level <= threshold;
}
}  // namespace mace

#endif  // MACE_PORT_LOGGER_H_
