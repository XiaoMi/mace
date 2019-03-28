// Copyright 2019 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_OPS_ARM_FP32_GEMV_H_
#define MACE_OPS_ARM_FP32_GEMV_H_

#include "mace/public/mace.h"
#include "mace/core/tensor.h"
#include "mace/core/op_context.h"

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

class Gemv {
 public:
  Gemv() {}
  ~Gemv() {}
  // Always row-major after transpose
  MaceStatus Compute(
      const OpContext *context,
      const Tensor *lhs,
      const Tensor *rhs,
      const Tensor *bias,
      const index_t batch,
      const index_t lhs_height,
      const index_t lhs_width,
      const bool lhs_batched,
      const bool rhs_batched,
      Tensor *output);
};

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_FP32_GEMV_H_
