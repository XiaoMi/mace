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

// This implements matrix-vector multiplication described as
// https://github.com/google/gemmlowp/blob/master/todo/fast-gemv.txt

#ifndef MACE_OPS_ARM_Q8_ELTWISE_H_
#define MACE_OPS_ARM_Q8_ELTWISE_H_

#include "mace/core/op_context.h"
#include "mace/core/types.h"
#include "mace/ops/common/eltwise_type.h"

namespace mace {
namespace ops {
namespace arm {
namespace q8 {

class Eltwise {
 public:
  explicit Eltwise(const EltwiseType type) : type_(type) {}

  MaceStatus Compute(const OpContext *context,
                     const Tensor *input0,
                     const Tensor *input1,
                     Tensor *output);

 private:
  EltwiseType type_;
};

}  // namespace q8
}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_Q8_ELTWISE_H_
