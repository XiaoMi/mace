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

#include "mace/utils/logging.h"

#include <stdlib.h>
#include <string.h>
#include <sstream>
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#include <iostream>
#endif

namespace mace {
namespace logging {

LogMessage::LogMessage(const char *fname, int line, int severity)
    : fname_(fname), line_(line), severity_(severity) {}

void LogMessage::DealWithFatal() {
  // When there is a fatal log, now we simply abort.
  abort();
}

void LogMessage::GenerateLogMessage() {
#if defined(ANDROID) || defined(__ANDROID__)
  int android_log_level;
  switch (severity_) {
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
      if (severity_ < INFO) {
        android_log_level = ANDROID_LOG_VERBOSE;
      } else {
        android_log_level = ANDROID_LOG_ERROR;
      }
      break;
  }

  std::stringstream ss;
  const char *const partial_name = strrchr(fname_, '/');
  ss << (partial_name != nullptr ? partial_name + 1 : fname_) << ":" << line_
     << " " << str();
  __android_log_write(android_log_level, "MACE", ss.str().c_str());

  // Also log to stderr (for standalone Android apps).
  std::cerr << "IWEF"[severity_] << " " << ss.str() << std::endl;
#else
  fprintf(stderr, "%c %s:%d] %s\n", "IWEF"[severity_], fname_, line_,
          str().c_str());
#endif

  // When there is a fatal log, terminate execution
  if (severity_ == FATAL) {
    DealWithFatal();
  }
}
namespace {

int LogLevelStrToInt(const char *mace_env_var_val) {
  if (mace_env_var_val == nullptr) {
    return 0;
  }
  // Simply use atoi here. Return 0 if convert unsuccessfully.
  return atoi(mace_env_var_val);
}

int MinLogLevelFromEnv() {
  // Read the min log level from env once during the first call to logging.
  static int log_level = LogLevelStrToInt(getenv("MACE_CPP_MIN_LOG_LEVEL"));
  return log_level;
}

int MinVLogLevelFromEnv() {
  // Read the min vlog level from env once during the first call to logging.
  static int vlog_level = LogLevelStrToInt(getenv("MACE_CPP_MIN_VLOG_LEVEL"));
  return vlog_level;
}

}  // namespace

LogMessage::~LogMessage() {
  int min_log_level = MinLogLevelFromEnv();
  if (severity_ >= min_log_level) GenerateLogMessage();
}

int LogMessage::MinVLogLevel() {
  return MinVLogLevelFromEnv();
}

}  // namespace logging
}  // namespace mace
