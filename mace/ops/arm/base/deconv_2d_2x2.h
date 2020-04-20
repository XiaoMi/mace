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

#ifndef MACE_OPS_ARM_BASE_DECONV_2D_2X2_H_
#define MACE_OPS_ARM_BASE_DECONV_2D_2X2_H_

#include <vector>
#include <memory>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/core/types.h"
#include "mace/ops/arm/base/deconv_2d_mxn.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/public/mace.h"

namespace mace {
namespace ops {
namespace arm {

template<typename T>
class Deconv2dK2x2S1 : public Deconv2dKMxN<T> {
 public:
  explicit Deconv2dK2x2S1(const delegator::Deconv2dParam &param)
      : Deconv2dKMxN<T>(param) {}
  virtual ~Deconv2dK2x2S1() {}

  MaceStatus DoCompute(const DeconvComputeParam &p, const T *filter,
                       const T *input_data, T *padded_out_data) override;
};

template<typename T>
class Deconv2dK2x2S2 : public Deconv2dKMxN<T> {
 public:
  explicit Deconv2dK2x2S2(const delegator::Deconv2dParam &param)
      : Deconv2dKMxN<T>(param) {}
  virtual ~Deconv2dK2x2S2() {}

  MaceStatus DoCompute(const DeconvComputeParam &p, const T *filter,
                       const T *input_data, T *padded_out_data) override;
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_DECONV_2D_2X2_H_
