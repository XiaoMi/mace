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


#ifndef MACE_OPS_REF_DEPTHWISE_DECONV_2D_H_
#define MACE_OPS_REF_DEPTHWISE_DECONV_2D_H_

#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/delegator/depthwise_deconv_2d.h"
#include "mace/public/mace.h"

namespace mace {
namespace ops {
namespace ref {

template<typename OUTPUT_TYPE>
class GroupDeconv2d : public delegator::GroupDeconv2d {
 public:
  explicit GroupDeconv2d(const delegator::GroupDeconv2dParam &param)
      : delegator::GroupDeconv2d(param) {}

  virtual ~GroupDeconv2d() = default;

  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *output_shape,
      Tensor *output) override;
};

template<typename OUTPUT_TYPE>
class DepthwiseDeconv2d : public GroupDeconv2d<OUTPUT_TYPE> {
 public:
  explicit DepthwiseDeconv2d(const delegator::DepthwiseDeconv2d &param)
      : GroupDeconv2d<OUTPUT_TYPE>(param) {}

  ~DepthwiseDeconv2d() = default;

  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *output_shape,
      Tensor *output) override;
};

template<>
class GroupDeconv2d<float> : public delegator::GroupDeconv2d {
 public:
  explicit GroupDeconv2d(const delegator::GroupDeconv2dParam &param)
      : delegator::GroupDeconv2d(param) {}

  virtual ~GroupDeconv2d() = default;

  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *output_shape,
      Tensor *output) override;
};

template<>
class DepthwiseDeconv2d<float> : public GroupDeconv2d<float> {
 public:
  explicit DepthwiseDeconv2d(const delegator::DepthwiseDeconv2dParam &param)
      : GroupDeconv2d(param) {}

  ~DepthwiseDeconv2d() = default;

  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *output_shape,
      Tensor *output) override;
};

}  // namespace ref
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_REF_DEPTHWISE_DECONV_2D_H_

