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

#include "mace/port/android/logger.h"

#include <android/log.h>

#include <iostream>

namespace mace {
namespace port {

void AndroidLogWriter::WriteLogMessage(const char *fname,
                                       const int line,
                                       const LogLevel severity,
                                       const char *message) {
  int android_log_level;
  switch (severity) {
    case INFO:
      android_log_level = ANDROID_LOG_INFO;
      break;
    case WARNING:
      android_log_level = ANDROID_LOG_WARN;
      break;
    case ERROR:
      android_log_level = ANDROID_LOG_ERROR;
      break;
    case FATAL:
      android_log_level = ANDROID_LOG_FATAL;
      break;
    default:
      android_log_level = ANDROID_LOG_ERROR;
      break;
  }

  std::stringstream ss;
  const char *const partial_name = strrchr(fname, '/');
  ss << (partial_name != nullptr ? partial_name + 1 : fname) << ":" << line
     << " " << message;
  __android_log_write(android_log_level, "MACE", ss.str().c_str());

  // Also log to stderr (for standalone Android apps) and abort.
  LogWriter::WriteLogMessage(fname, line, severity, message);
}

}  // namespace port
}  // namespace mace
