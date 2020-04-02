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

#ifndef MICRO_OPS_UTILS_GEMV_H_
#define MICRO_OPS_UTILS_GEMV_H_

#include "micro/base/types.h"
#include "micro/include/public/micro.h"

namespace micro {
namespace ops {

template<typename T>
class Gemv {
 public:
  Gemv() {}
  ~Gemv() {}
  // Always row-major after transpose
  MaceStatus Compute(
      const T *lhs_data,
      const T *rhs_data,
      const T *bias_data,
      const int32_t batch,
      const int32_t lhs_height,
      const int32_t lhs_width,
      const bool lhs_batched,
      const bool rhs_batched,
      T *output_data);
};

template<>
class Gemv<mifloat> {
 public:
  Gemv() {}
  ~Gemv() {}
  // Always row-major after transpose
  MaceStatus Compute(
      const mifloat *lhs_data,
      const mifloat *rhs_data,
      const mifloat *bias_data,
      const int32_t batch,
      const int32_t lhs_height,
      const int32_t lhs_width,
      const bool lhs_batched,
      const bool rhs_batched,
      mifloat *output_data);
};

}  // namespace ops
}  // namespace micro


#endif  // MICRO_OPS_UTILS_GEMV_H_
