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
#ifndef MACE_OPS_OPENCL_BUFFER_CONV_2D_H_
#define MACE_OPS_OPENCL_BUFFER_CONV_2D_H_

#include "mace/ops/opencl/conv_2d.h"

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
namespace conv2d {

extern MaceStatus Conv2d1x1(OpContext *context,
                            cl::Kernel *kernel,
                            const Tensor *padded_input,
                            const Tensor *filter,
                            const Tensor *bias,
                            const int *strides,
                            const ActivationType activation,
                            const float relux_max_limit,
                            const float leakyrelu_coefficient,
                            const bool input_changed,
                            Tensor *output,
                            StatsFuture *future);

extern MaceStatus Conv2dGeneral(OpContext *context,
                                cl::Kernel *kernel,
                                const Tensor *input,
                                const Tensor *filter,
                                const Tensor *bias,
                                const int *strides,
                                const int *dilations,
                                const ActivationType activation,
                                const float relux_max_limit,
                                const float leakyrelu_coefficient,
                                const bool input_changed,
                                Tensor *output,
                                StatsFuture *future);
}  // namespace conv2d

class Conv2dKernel : public OpenCLConv2dKernel {
 public:
  Conv2dKernel() : old_scratch_size_(0) {}

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
      const int winograd_blk_size,
      Tensor *output) override;

 private:
  index_t old_scratch_size_;
  cl::Kernel kernels_[2];
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_BUFFER_CONV_2D_H_
