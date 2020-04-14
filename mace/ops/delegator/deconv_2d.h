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


#ifndef MACE_OPS_DELEGATOR_DECONV_2D_H_
#define MACE_OPS_DELEGATOR_DECONV_2D_H_

#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/ops/op_delegator.h"
#include "mace/core/registry/op_delegator_registry.h"
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {
namespace ops {

enum DeconvType {
  K2x2S1,
  K2x2S2,
  K3x3S1,
  K3x3S2,
  K4x4S1,
  K4x4S2,
};

namespace delegator {

struct Deconv2dParam : public DelegatorParam {
  explicit Deconv2dParam(const std::vector<int> &strides,
                         const std::vector<int> &dilations,
                         const std::vector<int> &paddings,
                         const Padding padding_type,
                         const FrameworkType framework_type,
                         const int group = 1)
      : strides_(strides), dilations_(dilations),
        paddings_(paddings), padding_type_(padding_type),
        framework_type_(framework_type),
        group_(group) {}

  const std::vector<int> &strides_;
  const std::vector<int> &dilations_;
  const std::vector<int> &paddings_;
  const Padding padding_type_;
  const FrameworkType framework_type_;
  const int group_;
};

class Deconv2d : public OpDelegator {
 public:
  explicit Deconv2d(const Deconv2dParam &param)
      : OpDelegator(param),
        strides_(param.strides_),
        dilations_(param.dilations_),
        paddings_(param.paddings_),
        padding_type_(param.padding_type_),
        framework_type_(param.framework_type_),
        group_(param.group_) {}

  virtual ~Deconv2d() = default;

  MACE_DEFINE_DELEGATOR_CREATOR(Deconv2d)

  virtual MaceStatus Compute(const OpContext *context,
                             const Tensor *input,
                             const Tensor *filter,
                             const Tensor *output_shape,
                             Tensor *output) = 0;

 protected:
  const std::vector<int> strides_;
  const std::vector<int> dilations_;
  const std::vector<int> paddings_;
  const Padding padding_type_;
  const FrameworkType framework_type_;
  const int group_;
};

}  // namespace delegator
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_DELEGATOR_DECONV_2D_H_

