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


#include "micro/base/value_to_str.h"

namespace micro {
namespace base {

#ifndef MACE_SIGNED_TO_STRING
#define MACE_SIGNED_TO_STRING(T, UNSIGNED_T)                    \
template<>                                                      \
char *ToString(T value, char *buffer, char *end) {              \
  if (value < 0) {                                              \
    value = -value;                                             \
    *buffer++ = '-';                                            \
  }                                                             \
  return ToString(static_cast<UNSIGNED_T>(value), buffer, end); \
}
#endif  // MACE_SIGNED_TO_STRING


void ReverseInplace(char *start, char *end) {
  end--;
  while (start < end) {
    char tmp = *start;
    *start++ = *end;
    *end-- = tmp;
  }
}

MACE_SIGNED_TO_STRING(int64_t, uint64_t)

MACE_SIGNED_TO_STRING(int32_t, uint32_t)

MACE_SIGNED_TO_STRING(int16_t, uint16_t)

MACE_SIGNED_TO_STRING(int8_t, uint8_t)

template<>
char *ToString(const char *str, char *buffer, char *end) {
  end--;
  while (*str != '\0' && buffer < end) {
    *buffer++ = *str++;
  }
  *buffer = '\0';
  return buffer;
}

template<>
char *ToString(float value, char *buffer, char *end) {
  if (value <= -1e-8) {
    *buffer++ = '-';
  }
  int32_t int_part = (int32_t) value;
  buffer = ToString(int_part, buffer, end);

  float deci_part = value - int_part;
  if (deci_part < 1e-8 && deci_part > -1e-8) {
    return buffer;
  }
  if (deci_part < 0.0) {
    deci_part = -deci_part;
  }

  end--;
  *buffer++ = '.';
  do {
    deci_part *= 10;
    int32_t remainder = (int32_t) deci_part;
    *buffer++ = '0' + remainder;
    deci_part -= remainder;
  } while (deci_part > 0 && buffer < end);

  *buffer = '\0';
  return buffer;
}

}  // namespace base
}  // namespace micro
