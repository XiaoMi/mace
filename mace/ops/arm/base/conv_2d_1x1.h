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

#ifndef MACE_OPS_ARM_BASE_CONV_2D_1X1_H_
#define MACE_OPS_ARM_BASE_CONV_2D_1X1_H_

#include "mace/ops/arm/base/conv_2d.h"
#include "mace/ops/arm/base/gemm.h"

namespace mace {
namespace ops {
namespace arm {

template<typename T>
class Conv2dK1x1 : public Conv2dBase {
 public:
  explicit Conv2dK1x1(const delegator::Conv2dParam &param)
      : Conv2dBase(param, sizeof(T)),
        gemm_(delegator::GemmParam()) {}
  virtual ~Conv2dK1x1() {}

  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      Tensor *output) override;

 private:
  Gemm<T> gemm_;
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_CONV_2D_1X1_H_
