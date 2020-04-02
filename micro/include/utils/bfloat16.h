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

#ifndef MICRO_INCLUDE_UTILS_BFLOAT16_H_
#define MICRO_INCLUDE_UTILS_BFLOAT16_H_

#include <stdint.h>

#ifdef MACE_ENABLE_BFLOAT16

namespace micro {

union Sphinx {
  uint32_t i;
  float f;

  Sphinx(uint32_t value) : i(value) {}

  Sphinx(float value) : f(value) {}
};

class BFloat16 {
 public:
  BFloat16();

  operator float() const {
    return Sphinx(static_cast<uint32_t>(data_ << 16)).f;
  }

  void operator=(const BFloat16 &value) {
    data_ = value.data_;
  }

  void operator=(float value) {
    data_ = Sphinx(value).i >> 16;
  }

 public:
  uint16_t data_;
};

}  // namespace micro

#endif  // MACE_ENABLE_BFLOAT16

#endif  // MICRO_INCLUDE_UTILS_BFLOAT16_H_
