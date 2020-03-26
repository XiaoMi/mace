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
#ifndef MACE_OPS_OPENCL_BUFFER_DEPTHWISE_CONV2D_H_
#define MACE_OPS_OPENCL_BUFFER_DEPTHWISE_CONV2D_H_

#include "mace/ops/opencl/depthwise_conv2d.h"

#include <functional>
#include <memory>
#include <vector>

#include "mace/ops/opencl/buffer/utils.h"
#include "mace/core/runtime/opencl/opencl_helper.h"
#include "mace/utils/memory.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {
namespace depthwise {

MaceStatus DepthwiseConv2d(OpContext *context,
                           cl::Kernel *kernel,
                           const Tensor *padded_input,   // NHWC
                           const Tensor *filter,  // HWIM
                           const Tensor *bias,
                           const int *strides,
                           const int *dilations,
                           const ActivationType activation,
                           const float relux_max_limit,
                           const float leakyrelu_coefficient,
                           const bool input_changed,
                           Tensor *output,
                           StatsFuture *future);
}  // namespace depthwise

class DepthwiseConv2dKernel : public OpenCLDepthwiseConv2dKernel {
 public:
  DepthwiseConv2dKernel() : old_scratch_size_(0) {}
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
      Tensor *output) override;

 private:
  index_t old_scratch_size_;
  cl::Kernel kernels_[2];
  std::vector<index_t> input_shape_;
};

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_BUFFER_DEPTHWISE_CONV2D_H_
