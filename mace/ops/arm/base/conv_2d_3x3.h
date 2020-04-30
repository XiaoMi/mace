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

#ifndef MACE_OPS_ARM_BASE_CONV_2D_3X3_H_
#define MACE_OPS_ARM_BASE_CONV_2D_3X3_H_

#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/arm/base/conv_2d_mxn.h"
#include "mace/public/mace.h"

namespace mace {
namespace ops {
namespace arm {

template<typename T>
class Conv2dK3x3S1 : public Conv2dKMxN<T> {
 public:
  explicit Conv2dK3x3S1(const delegator::Conv2dParam &param)
      : Conv2dKMxN<T>(param, 2, 4) {}
  virtual ~Conv2dK3x3S1() {}

  MaceStatus DoCompute(const ConvComputeParam &p, const T *filter,
                       const T *input_data, T *output_data) override;
};

template<typename T>
class Conv2dK3x3S2 : public Conv2dKMxN<T> {
 public:
  explicit Conv2dK3x3S2(const delegator::Conv2dParam &param)
      : Conv2dKMxN<T>(param, 1, 4) {}
  virtual ~Conv2dK3x3S2() {}

  MaceStatus DoCompute(const ConvComputeParam &p, const T *filter,
                       const T *input_data, T *output_data) override;
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_CONV_2D_3X3_H_
