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

#ifndef MACE_OPS_ARM_FP32_CONV_2D_H_
#define MACE_OPS_ARM_FP32_CONV_2D_H_

#include <vector>
#include <memory>

#include "mace/public/mace.h"
#include "mace/core/tensor.h"
#include "mace/core/op_context.h"
#include "mace/ops/arm/fp32/gemm.h"
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

class Conv2dBase {
 public:
  Conv2dBase(const std::vector<int> &strides,
             const std::vector<int> &dilations,
             const std::vector<int> &paddings,
             const Padding padding_type)
      : strides_(strides),
        dilations_(dilations),
        paddings_(paddings),
        padding_type_(padding_type) {}

  virtual ~Conv2dBase() = default;

  virtual MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      Tensor *output) = 0;

 protected:
  void CalOutputShapeAndInputPadSize(const std::vector<index_t> &input_shape,
                                     const std::vector<index_t> &filter_shape,
                                     std::vector<index_t> *output_shape,
                                     std::vector<int> *in_pad_size);

  void CalOutputBoundaryWithoutUsingInputPad(const std::vector<index_t>
                                             &output_shape,
                                             const std::vector<int>
                                             in_pad_size,
                                             std::vector<index_t>
                                             *out_bound);

  void CalOutputShapeAndPadSize(const Tensor *input,
                                const Tensor *filter,
                                const int out_tile_height,
                                const int out_tile_width,
                                std::vector<index_t> *output_shape,
                                std::vector<int> *in_pad_size,
                                std::vector<int> *out_pad_size);

  MaceStatus ResizeOutAndPadInOut(const OpContext *context,
                                  const Tensor *input,
                                  const Tensor *filter,
                                  Tensor *output,
                                  const int out_tile_height,
                                  const int out_tile_width,
                                  std::unique_ptr<const Tensor> *padded_input,
                                  std::unique_ptr<Tensor> *padded_output);

  void PadInput(const Tensor &src,
                const int pad_top,
                const int pad_left,
                Tensor *dst);
  void UnPadOutput(const Tensor &src, Tensor *dst);

  const std::vector<int> strides_;
  const std::vector<int> dilations_;
  const std::vector<int> paddings_;
  const Padding padding_type_;
};

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_FP32_CONV_2D_H_
