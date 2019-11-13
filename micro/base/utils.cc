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


#include "micro/base/utils.h"

#include <math.h>

#include "micro/base/logging.h"

namespace micro {
namespace base {

uint32_t strlen(const char *str) {
  MACE_ASSERT1(str != NULL, "str can not be NULL.");
  uint32_t length = 0;
  while (*str++ != '\0') {
    ++length;
  }
  return length;
}

int32_t strcmp(const char *str1, const char *str2) {
  MACE_ASSERT1(str1 != NULL && str2 != NULL,
               "strcmp str can not be NULL.");
  while (*str1 == *str2) {
    if (*str1 == '\0') {
      return 0;
    }
    ++str1;
    ++str2;
  }
  return (*str1) - (*str2);
}

void memcpy(void *dst, const void *src, uint32_t bytes) {
  MACE_ASSERT1(dst != NULL && src != NULL && bytes > 0,
               "Invalid params.");
  uint8_t *dst_mem = static_cast<uint8_t *>(dst);
  const uint8_t *src_mem = static_cast<const uint8_t *>(src);
  while (bytes-- > 0) {
    *dst_mem++ = *src_mem++;
  }
}

int32_t GetShapeSize(uint32_t dim_size, const int32_t *dims) {
  return accumulate_multi(dims, 0, dim_size);
}

float sqrt(float x) {
  return ::sqrt(x);
}

int32_t ceil(float f) {
  int32_t i = (int32_t) f;
  return (f == static_cast<float>(i)) ? i : i + 1;
}

int32_t floor(float f) {
  return ::floor(f);
}

float fabs(float x) {
  if (x < 0.0f) {
    return -x;
  } else if (x > 0.0f) {
    return x;
  } else {
    return 0.0f;
  }
}

float lowest() {
  return -3.402823466e+38F;
}

float highest() {
  return 3.402823466e+38F;
}

float tanh(float x) {
  return ::tanh(x);
}

float exp(float x) {
  return ::exp(x);
}

float pow(float x, float y) {
  return ::pow(x, y);
}

float log(float x) {
  return ::log(x);
}

}  // namespace base
}  // namespace micro
