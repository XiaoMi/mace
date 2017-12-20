//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/utils/logging.h"

#include <stdlib.h>
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#include <iostream>
#include <sstream>
#endif

namespace mace {
namespace internal {

LogMessage::LogMessage(const char *fname, int line, int severity)
    : fname_(fname), line_(line), severity_(severity) {}

#if defined(ANDROID) || defined(__ANDROID__)
void LogMessage::GenerateLogMessage() {
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

  // Android logging at level FATAL does not terminate execution, so abort()
  // is still required to stop the program.
  if (severity_ == FATAL) {
    abort();
  }
}

#else

void LogMessage::GenerateLogMessage() {
  fprintf(stderr, "%c %s:%d] %s\n", "IWEF"[severity_], fname_, line_,
          str().c_str());
}
#endif

namespace {

// Parse log level (int64_t) from environment variable (char*)
int64_t LogLevelStrToInt(const char *mace_env_var_val) {
  if (mace_env_var_val == nullptr) {
    return 0;
  }

  // Ideally we would use env_var / safe_strto64, but it is
  // hard to use here without pulling in a lot of dependencies,
  // so we use std:istringstream instead
  string min_log_level(mace_env_var_val);
  std::istringstream ss(min_log_level);
  int64_t level;
  if (!(ss >> level)) {
    // Invalid vlog level setting, set level to default (0)
    level = 0;
  }

  return level;
}

int64_t MinLogLevelFromEnv() {
  const char *mace_env_var_val = getenv("MACE_CPP_MIN_LOG_LEVEL");
  return LogLevelStrToInt(mace_env_var_val);
}

int64_t MinVLogLevelFromEnv() {
  const char *mace_env_var_val = getenv("MACE_CPP_MIN_VLOG_LEVEL");
  return LogLevelStrToInt(mace_env_var_val);
}

}  // namespace

LogMessage::~LogMessage() {
  // Read the min log level once during the first call to logging.
  static int64_t min_log_level = MinLogLevelFromEnv();
  if (severity_ >= min_log_level) GenerateLogMessage();
}

int64_t LogMessage::MinVLogLevel() {
  static int64_t min_vlog_level = MinVLogLevelFromEnv();
  return min_vlog_level;
}

LogMessageFatal::LogMessageFatal(const char *file, int line)
    : LogMessage(file, line, FATAL) {}
LogMessageFatal::~LogMessageFatal() {
  // abort() ensures we don't return (we promised we would not via
  // ATTRIBUTE_NORETURN).
  GenerateLogMessage();
  abort();
}

}  // namespace internal
}  // namespace mace
