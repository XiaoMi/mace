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

#ifndef MACE_OPS_ARM_FP32_DECONV_2D_H_
#define MACE_OPS_ARM_FP32_DECONV_2D_H_

#include <vector>
#include <memory>

#include "mace/public/mace.h"
#include "mace/core/tensor.h"
#include "mace/core/types.h"
#include "mace/core/op_context.h"
#include "mace/ops/arm/fp32/gemm.h"
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

class Deconv2dBase {
 public:
  Deconv2dBase(const std::vector<int> &strides,
               const std::vector<int> &dilations,
               const std::vector<int> &paddings,
               const Padding padding_type,
               const index_t group,
               const FrameworkType framework_type)
      : strides_(strides),
        dilations_(dilations),
        paddings_(paddings),
        padding_type_(padding_type),
        group_(group),
        framework_type_(framework_type) {}

  Deconv2dBase(const std::vector<int> &strides,
               const std::vector<int> &dilations,
               const std::vector<int> &paddings,
               const Padding padding_type,
               const FrameworkType framework_type)
      : Deconv2dBase(strides,
                     dilations,
                     paddings,
                     padding_type,
                     1,
                     framework_type) {}

  virtual ~Deconv2dBase() = default;

  virtual MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *output_shape,
      Tensor *output) = 0;

 protected:
  MaceStatus ResizeOutAndPadOut(const OpContext *context,
                                const Tensor *input,
                                const Tensor *filter,
                                const Tensor *output_shape,
                                Tensor *output,
                                std::vector<int> *out_pad_size,
                                std::unique_ptr<Tensor> *padded_output);

  void UnPadOutput(const Tensor &src,
                   const std::vector<int> &out_pad_size,
                   Tensor *dst);

  const std::vector<int> strides_;
  const std::vector<int> dilations_;
  const std::vector<int> paddings_;
  const Padding padding_type_;
  index_t group_;
  const FrameworkType framework_type_;
};

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_FP32_DECONV_2D_H_
