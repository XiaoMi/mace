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

#ifndef MACE_OPS_ARM_BASE_DEPTHWISE_DECONV_2D_3X3_H_
#define MACE_OPS_ARM_BASE_DEPTHWISE_DECONV_2D_3X3_H_

#include <vector>
#include <memory>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/core/types.h"
#include "mace/ops/arm/base/depthwise_deconv_2d_mxn.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/delegator/depthwise_deconv_2d.h"
#include "mace/public/mace.h"

namespace mace {
namespace ops {
namespace arm {

template<typename T>
class DepthwiseDeconv2dK3x3S1 : public DepthwiseDeconv2dKMxN<T> {
 public:
  explicit DepthwiseDeconv2dK3x3S1(
      const delegator::DepthwiseDeconv2dParam &param)
      : DepthwiseDeconv2dKMxN<T>(param) {}
  virtual ~DepthwiseDeconv2dK3x3S1() {}

  MaceStatus DoCompute(const DepthwiseDeconvComputeParam &p, const T *filter,
                       const T *input_data, T *padded_out_data) override;
};

template<typename T>
class DepthwiseDeconv2dK3x3S2 : public DepthwiseDeconv2dKMxN<T> {
 public:
  explicit DepthwiseDeconv2dK3x3S2(
      const delegator::DepthwiseDeconv2dParam &param)
      : DepthwiseDeconv2dKMxN<T>(param) {}
  virtual ~DepthwiseDeconv2dK3x3S2() {}

  MaceStatus DoCompute(const DepthwiseDeconvComputeParam &p, const T *filter,
                       const T *input_data, T *padded_out_data) override;
};

template<typename T>
class GroupDeconv2dK3x3S1 : public GroupDeconv2dKMxN<T> {
 public:
  explicit GroupDeconv2dK3x3S1(
      const delegator::GroupDeconv2dParam &param)
      : GroupDeconv2dKMxN<T>(param) {}
  virtual ~GroupDeconv2dK3x3S1() {}

  MaceStatus DoCompute(const GroupDeconvComputeParam &p, const T *filter,
                       const T *input_data, T *padded_out_data) override;
};

template<typename T>
class GroupDeconv2dK3x3S2 : public GroupDeconv2dKMxN<T> {
 public:
  explicit GroupDeconv2dK3x3S2(const delegator::GroupDeconv2dParam &param)
      : GroupDeconv2dKMxN<T>(param) {}
  virtual ~GroupDeconv2dK3x3S2() {}

  MaceStatus DoCompute(const GroupDeconvComputeParam &p, const T *filter,
                       const T *input_data, T *padded_out_data) override;
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_DEPTHWISE_DECONV_2D_3X3_H_
