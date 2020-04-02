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

#ifndef MICRO_BASE_LOGGER_H_
#define MICRO_BASE_LOGGER_H_

#include <stdint.h>

namespace micro {

enum LogLevel {
  CLEAN = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3,
  FATAL = 4,
  INVALID_MAX,
};

namespace base {

class Logger {
 public:
  Logger(const char *fname, uint32_t line, LogLevel severity);
  ~Logger();

  const Logger &operator<<(const char *str) const;
  const Logger &operator<<(const char c) const;
  const Logger &operator<<(const float value) const;
  const Logger &operator<<(const int64_t value) const;
  const Logger &operator<<(const int32_t value) const;
  const Logger &operator<<(const uint32_t value) const;
  const Logger &operator<<(const int16_t value) const;
  const Logger &operator<<(const uint16_t value) const;
  const Logger &operator<<(const int8_t value) const;
  const Logger &operator<<(const uint8_t value) const;
  const Logger &operator<<(const bool value) const;

 private:
  LogLevel severity_;
};

}  // namespace base
}  // namespace micro

#endif  // MICRO_BASE_LOGGER_H_
