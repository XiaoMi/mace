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

#ifndef MACE_OPS_OPENCL_DEPTHWISE_CONV2D_H_
#define MACE_OPS_OPENCL_DEPTHWISE_CONV2D_H_

#include <vector>

#include "mace/ops/common/activation_type.h"
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {

class OpContext;

namespace ops {
class OpenCLDepthwiseConv2dKernel {
 public:
  virtual MaceStatus Compute(
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
      Tensor *output) = 0;
  MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLDepthwiseConv2dKernel);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_DEPTHWISE_CONV2D_H_
