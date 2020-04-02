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

#ifndef MICRO_BASE_LOGGING_H_
#define MICRO_BASE_LOGGING_H_

#include <stdint.h>

#include "micro/base/logger.h"
#include "micro/include/port/define.h"

namespace micro {
namespace log {

#define LOG(severity) \
  micro::base::Logger(__FILE__, __LINE__, micro::severity)

#ifndef NDEBUG
#define LOG1(severity, value) LOG(severity) << value
#define LOG2(severity, value1, value2) LOG(severity) << value1 << value2
#define LOG3(severity, value1, value2, value3) \
  LOG(severity) << value1 << value2 << value3
#define LOG4(severity, value1, value2, value3, value4) \
  LOG(severity) << value1 << value2 << value3 << value4
#define LOG5(severity, value1, value2, value3, value4, value5) \
  LOG(severity) << value1 << value2 << value3 << value4 << value5
#else
#define LOG1(severity, value)
#define LOG2(severity, value1, value2)
#define LOG3(severity, value1, value2, value3)
#define LOG4(severity, value1, value2, value3, value4)
#define LOG5(severity, value1, value2, value3, value4, value5)
#endif  // NDEBUG

#ifndef NDEBUG
#define MACE_ASSERT(condition) \
  if (!(condition)) LOG(FATAL) << "Assert failed: "#condition  // NOLINT
#define MACE_ASSERT1(condition, str) \
  if (!(condition)) LOG(FATAL) << "Assert failed: "#condition " " << str   // NOLINT
#define MACE_ASSERT2(condition, str1, str2) \
  if (!(condition)) LOG(FATAL) << "Assert failed: "#condition " " << str1 << str2  // NOLINT
#else
#define MACE_ASSERT(condition)
#define MACE_ASSERT1(condition, string)
#define MACE_ASSERT2(condition, string1, string2)
#endif  // NDEBUG

#define MACE_NOT_IMPLEMENTED MACE_ASSERT1(false, "not implemented")

#define MACE_CHECK_SUCCESS(stmt)                    \
  {                                                 \
    MaceStatus status = (stmt);                     \
    if (status != MACE_SUCCESS) {                   \
      LOG(FATAL) << #stmt << " failed with error: " \
              << status;                            \
    }                                               \
  }

#define MACE_RETURN_IF_ERROR(stmt)               \
  {                                              \
    MaceStatus status = (stmt);                  \
    if (status != MACE_SUCCESS) {                \
      LOG(INFO) << static_cast<int32_t>(stmt)    \
                << " failed with error: "        \
                << static_cast<int32_t>(status); \
      return status;                             \
    }                                            \
  }

}  // namespace log
}  // namespace micro

#endif  // MICRO_BASE_LOGGING_H_
