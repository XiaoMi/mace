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
#ifndef MACE_OPS_OPENCL_IMAGE_ACTIVATION_H_
#define MACE_OPS_OPENCL_IMAGE_ACTIVATION_H_

#include "mace/ops/opencl/activation.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/common/activation_type.h"
#include "mace/core/runtime/opencl/opencl_helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

class ActivationKernel : public OpenCLActivationKernel {
 public:
  ActivationKernel(ActivationType type,
                   float relux_max_limit,
                   float leakyrelu_coefficient)
      : activation_(type), relux_max_limit_(relux_max_limit),
        leakyrelu_coefficient_(leakyrelu_coefficient) {}

  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *alpha,
      Tensor *output) override;

 private:
  ActivationType activation_;
  float relux_max_limit_;
  float leakyrelu_coefficient_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
  std::string tuning_key_prefix_;
};

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_ACTIVATION_H_
