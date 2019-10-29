// Copyright 2018 The MACE Authors. All Rights Reserved.
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
#ifndef MACE_OPS_OPENCL_IMAGE_CONV_2D_H_
#define MACE_OPS_OPENCL_IMAGE_CONV_2D_H_

#include "mace/ops/opencl/conv_2d.h"

#include <memory>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

extern MaceStatus Conv2dK1x1(OpContext *context,
                             cl::Kernel *kernel,
                             const Tensor *input,
                             const Tensor *filter,
                             const Tensor *bias,
                             const int stride_h,
                             const int stride_w,
                             const int *padding,
                             const int *dilations,
                             const ActivationType activation,
                             const float relux_max_limit,
                             const float leakyrelu_coefficient,
                             std::vector<index_t> *prev_input_shape,
                             Tensor *output,
                             uint32_t *kwg_size);

extern MaceStatus Conv2dK3x3(OpContext *context,
                             cl::Kernel *kernel,
                             const Tensor *input,
                             const Tensor *filter,
                             const Tensor *bias,
                             const int stride_h,
                             const int stride_w,
                             const int *padding,
                             const int *dilations,
                             const ActivationType activation,
                             const float relux_max_limit,
                             const float leakyrelu_coefficient,
                             std::vector<index_t> *prev_input_shape,
                             Tensor *output,
                             uint32_t *kwg_size);

extern MaceStatus Conv2d(OpContext *context,
                         cl::Kernel *kernel,
                         const Tensor *input,
                         const Tensor *filter,
                         const Tensor *bias,
                         const int stride_h,
                         const int stride_w,
                         const int *padding,
                         const int *dilations,
                         const ActivationType activation,
                         const float relux_max_limit,
                         const float leakyrelu_coefficient,
                         std::vector<index_t> *prev_input_shape,
                         Tensor *output,
                         uint32_t *kwg_size);

extern MaceStatus WinogradConv2dK3x3S1(OpContext *context,
                                       cl::Kernel *kernels[3],
                                       const Tensor *input,
                                       const Tensor *filter,
                                       const Tensor *bias,
                                       const int *padding,
                                       const ActivationType activation,
                                       const float relux_max_limit,
                                       const float leakyrelu_coefficient,
                                       const int wino_blk_size,
                                       std::vector<index_t> *prev_input_shape,
                                       Tensor *output,
                                       uint32_t *kwg_size[3]);

class Conv2dKernel : public OpenCLConv2dKernel {
 public:
  bool CheckUseWinograd(
      OpenCLRuntime *runtime,
      const std::vector<index_t> &filter_shape,
      const std::vector<index_t> &output_shape,
      const int *strides,
      const int *dilations,
      int *wino_block_size) override;

  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *bias,
      const int *strides,
      const Padding &padding_type,
      const std::vector<int> &padding_data,
      const int *dilations,
      const ActivationType activation,
      const float relux_max_limit,
      const float leakyrelu_coefficient,
      const int wino_blk_size,
      Tensor *output) override;

 private:
  cl::Kernel kernels_[3];
  uint32_t kwg_size_[3];
  std::vector<index_t> input_shape_;
};

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_CONV_2D_H_
