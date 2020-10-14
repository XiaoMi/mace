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

#ifndef MACE_CORE_FP16_H_
#define MACE_CORE_FP16_H_

#ifdef MACE_ENABLE_FP16

#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <sstream>


namespace std {
inline float fabs(const float16_t &value) {
  return fabs(static_cast<float>(value));
}

inline float abs(const float16_t &value) {
  return abs(static_cast<float>(value));
}

inline float sqrt(const float16_t &value) {
  return sqrt(static_cast<float>(value));
}

inline float log(const float16_t &value) {
  return log(static_cast<float>(value));
}

inline float tanh(const float16_t &value) {
  return tanh(static_cast<float>(value));
}

inline float exp(const float16_t &value) {
  return exp(static_cast<float>(value));
}

inline int ceil(const float16_t &value) {
  return ceil(static_cast<float>(value));
}

inline int floor(const float16_t &value) {
  return floor(static_cast<float>(value));
}

inline float max(const float16_t &a, const float &b) {
  return max(static_cast<float>(a), b);
}

inline float max(const float &a, const float16_t &b) {
  return max(a, static_cast<float>(b));
}

inline float min(const float16_t &a, const float &b) {
  return min(static_cast<float>(a), b);
}

inline float min(const float &a, const float16_t &b) {
  return min(a, static_cast<float>(b));
}

inline float pow(const float16_t &a, const float16_t &b) {
  return pow(static_cast<float>(a), static_cast<float>(b));
}

inline float pow(const float16_t &a, const float &b) {
  return pow(static_cast<float>(a), b);
}

inline float pow(const float &a, const float16_t &b) {
  return pow(a, static_cast<float>(b));
}

inline ostream &operator<<(ostream &ss,  // NOLINT
                           const float16_t &value) {
  return ss << static_cast<float>(value);
}

}  // namespace std


#endif  // MACE_ENABLE_FP16

#endif  // MACE_CORE_FP16_H_
