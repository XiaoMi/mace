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

#ifndef MICRO_BASE_UTILS_H_
#define MICRO_BASE_UTILS_H_

#include <stdint.h>

#include "micro/base/logging.h"

namespace micro {
namespace base {

uint32_t strlen(const char *str);
int32_t strcmp(const char *str1, const char *str2);
void memcpy(void *dst, const void *src, uint32_t bytes);
int32_t GetShapeSize(uint32_t dim_size, const int32_t *dims);
float sqrt(float x);
int32_t ceil(float f);
int32_t floor(float f);
float fabs(float x);
float lowest();
float highest();
float tanh(float x);
float exp(float x);
float pow(float x, float y);
float log(float x);

template<typename T>
void memset(T *src, T value, uint32_t size) {
  for (uint32_t i = 0; i < size; ++i) {
    src[i] = value;
  }
}

template<typename T>
T accumulate_multi(const T *array, uint32_t array_start, uint32_t array_end) {
  MACE_ASSERT(array_start >= 0 && array_start <= array_end);
  if (array == NULL || array_start == array_end) {
    return 1;
  }
  T total = array[array_start];
  for (uint32_t i = array_start + 1; i < array_end; ++i) {
    total *= array[i];
  }
  return total;
}

template<typename T>
T abs(T x) {
  return x > 0 ? x : -x;
}

template<typename T>
T max(T a, T b) {
  return a > b ? a : b;
}

template<typename T>
T min(T a, T b) {
  return a < b ? a : b;
}

template<typename T>
void swap(T *a, T *b) {  // NOLINT
  T c = *a;
  *a = *b;
  *b = c;
}

template<typename T>
T clamp(T in, T low, T high) {
  return max<T>(low, min<T>(in, high));  // NOLINT
}

}  // namespace base
}  // namespace micro

#endif  // MICRO_BASE_UTILS_H_
