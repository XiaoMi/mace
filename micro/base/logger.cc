// Copyright 2020 The MACE Authors. All Rights Reserved.
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



#include "micro/base/logger.h"

#include "micro/base/value_to_str.h"
#include "micro/port/api.h"

namespace micro {
namespace base {

namespace {
const int32_t kInt64ValueBufferLength = 21;
const int32_t kInt32ValueBufferLength = 12;
const int32_t kInt16ValueBufferLength = 6;
const int32_t kInt8ValueBufferLength = 4;
const int32_t kFloatValueBufferLength = 21;

inline bool IsValidLogLevel(const LogLevel level) {
  return level >= CLEAN && level < INVALID_MAX;
}

char LogLevelToShortStr(LogLevel level) {
  if (!IsValidLogLevel(level)) {
    level = INFO;
  }

  return "CIWEF"[static_cast<int>(level)];
}

}  // namespace

Logger::Logger(const char *fname, uint32_t line,
               LogLevel severity) : severity_(severity) {
  if (severity == CLEAN) {
    return;
  }
  char buffer[15] = {0};
  char *end = buffer + 15;
  buffer[0] = LogLevelToShortStr(severity);
  buffer[1] = ' ';
  micro::port::api::DebugLog(buffer);

  micro::port::api::DebugLog(fname);

  buffer[0] = ':';
  ToString("] ", ToString(line, buffer + 1, end), end);
  micro::port::api::DebugLog(buffer);
}

Logger::~Logger() {
  micro::port::api::DebugLog("\n");
  if (severity_ == FATAL) {
    micro::port::api::Abort();
  }
}

const Logger &Logger::operator<<(const char *str) const {
  micro::port::api::DebugLog(str);
  return *this;
}

const Logger &Logger::operator<<(const char c) const {
  char buffer[2] = {0};
  buffer[0] = c;
  micro::port::api::DebugLog(buffer);
  return *this;
}

const Logger &Logger::operator<<(const float value) const {
  char buffer[kFloatValueBufferLength] = {0};
  ToString(value, buffer, buffer + kFloatValueBufferLength);
  micro::port::api::DebugLog(buffer);
  return *this;
}

const Logger &Logger::operator<<(const int64_t value) const {
  char buffer[kInt64ValueBufferLength] = {0};
  ToString(value, buffer, buffer + kInt64ValueBufferLength);
  micro::port::api::DebugLog(buffer);
  return *this;
}

const Logger &Logger::operator<<(const int32_t value) const {
  char buffer[kInt32ValueBufferLength] = {0};
  ToString(value, buffer, buffer + kInt32ValueBufferLength);
  micro::port::api::DebugLog(buffer);
  return *this;
}

const Logger &Logger::operator<<(const uint32_t value) const {
  char buffer[kInt32ValueBufferLength] = {0};
  ToString(value, buffer, buffer + kInt32ValueBufferLength);
  micro::port::api::DebugLog(buffer);
  return *this;
}

const Logger &Logger::operator<<(const int16_t value) const {
  char buffer[kInt16ValueBufferLength] = {0};
  ToString(value, buffer, buffer + kInt16ValueBufferLength);
  micro::port::api::DebugLog(buffer);
  return *this;
}

const Logger &Logger::operator<<(const uint16_t value) const {
  char buffer[kInt16ValueBufferLength] = {0};
  ToString(value, buffer, buffer + kInt16ValueBufferLength);
  micro::port::api::DebugLog(buffer);
  return *this;
}

const Logger &Logger::operator<<(const int8_t value) const {
  char buffer[kInt8ValueBufferLength] = {0};
  ToString(value, buffer, buffer + kInt8ValueBufferLength);
  micro::port::api::DebugLog(buffer);
  return *this;
}

const Logger &Logger::operator<<(const uint8_t value) const {
  char buffer[kInt8ValueBufferLength] = {0};
  ToString(value, buffer, buffer + kInt8ValueBufferLength);
  micro::port::api::DebugLog(buffer);
  return *this;
}

const Logger &Logger::operator<<(const bool value) const {
  if (value) {
    micro::port::api::DebugLog("true");
  } else {
    micro::port::api::DebugLog("false");
  }
  return *this;
}

}  // namespace base
}  // namespace micro
