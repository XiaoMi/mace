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


#ifndef MACE_OPS_REF_DECONV_2D_H_
#define MACE_OPS_REF_DECONV_2D_H_

#include <vector>

#include "mace/public/mace.h"
#include "mace/core/tensor.h"
#include "mace/core/op_context.h"
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {
namespace ops {
namespace ref {

template<typename OUTPUT_TYPE>
class Deconv2d {
 public:
  Deconv2d(const std::vector<int> &strides,
           const std::vector<int> &dilations,
           const std::vector<int> &paddings,
           const Padding padding_type,
           const FrameworkType framework_type)
      : strides_(strides),
        dilations_(dilations),
        paddings_(paddings),
        padding_type_(padding_type),
        framework_type_(framework_type) {}

  ~Deconv2d() = default;

  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *output_shape,
      Tensor *output);

 private:
  const std::vector<int> strides_;
  const std::vector<int> dilations_;
  const std::vector<int> paddings_;
  const Padding padding_type_;
  const FrameworkType framework_type_;
};

template<>
class Deconv2d<float> {
 public:
  Deconv2d(const std::vector<int> &strides,
           const std::vector<int> &dilations,
           const std::vector<int> &paddings,
           const Padding padding_type,
           const FrameworkType framework_type)
      : strides_(strides),
        dilations_(dilations),
        paddings_(paddings),
        padding_type_(padding_type),
        framework_type_(framework_type) {}

  ~Deconv2d() = default;

  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *output_shape,
      Tensor *output);

 private:
  const std::vector<int> strides_;
  const std::vector<int> dilations_;
  const std::vector<int> paddings_;
  const Padding padding_type_;
  const FrameworkType framework_type_;
};

}  // namespace ref
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_REF_DECONV_2D_H_

