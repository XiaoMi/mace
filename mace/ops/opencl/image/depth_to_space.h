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
#ifndef MACE_OPS_OPENCL_IMAGE_DEPTH_TO_SPACE_H_
#define MACE_OPS_OPENCL_IMAGE_DEPTH_TO_SPACE_H_

#include "mace/ops/opencl/depth_to_space.h"

#include <memory>
#include <vector>
#include <set>
#include <string>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/runtimes/opencl/core/opencl_helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

class DepthToSpaceKernel : public OpenCLDepthToSpaceKernel {
 public:
  explicit DepthToSpaceKernel(const int block_size, const std::string mode)
      : block_size_(block_size), mode_(mode) {}
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      Tensor *output) override;

 private:
  const int block_size_;
  const std::string mode_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_DEPTH_TO_SPACE_H_
