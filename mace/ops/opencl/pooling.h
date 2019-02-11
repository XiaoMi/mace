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

#ifndef MACE_OPS_OPENCL_POOLING_H_
#define MACE_OPS_OPENCL_POOLING_H_

#include <vector>

#include "mace/ops/pooling.h"
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {

class OpContext;
class Tensor;
namespace ops {
class OpenCLPoolingKernel {
 public:
  virtual MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const PoolingType pooling_type,
      const int *kernels,
      const int *strides,
      const Padding &padding_type,
      const std::vector<int> &padding_data,
      const int *dilations,
      const RoundType round_type,
      Tensor *output) = 0;
  MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLPoolingKernel);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_POOLING_H_
