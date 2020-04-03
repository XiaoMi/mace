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
#ifndef MACE_OPS_OPENCL_IMAGE_ELTWISE_H_
#define MACE_OPS_OPENCL_IMAGE_ELTWISE_H_

#include "mace/ops/opencl/eltwise.h"

#include <memory>
#include <utility>
#include <vector>
#include <set>
#include <string>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/common/eltwise_type.h"
#include "mace/core/runtime/opencl/opencl_helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

class EltwiseKernel : public OpenCLEltwiseKernel {
 public:
  explicit EltwiseKernel(
      const EltwiseType type,
      const std::vector<float> &coeff,
      const float scalar_input,
      const int32_t scalar_input_index)
      : type_(type),
        coeff_(coeff),
        scalar_input_(scalar_input),
        scalar_input_index_(scalar_input_index) {}
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input0,
      const Tensor *input1,
      Tensor *output) override;

 private:
  EltwiseType type_;
  std::vector<float> coeff_;
  float scalar_input_;
  int32_t scalar_input_index_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_ELTWISE_H_
