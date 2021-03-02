// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_RUNTIMES_OPENCL_TRANSFORM_BUFFER_TRANSFORM_KERNEL_H_
#define MACE_RUNTIMES_OPENCL_TRANSFORM_BUFFER_TRANSFORM_KERNEL_H_

#include "mace/public/mace.h"
#include "mace/runtimes/opencl/core/opencl_util.h"
#include "mace/utils/math.h"

namespace mace {
class OpContext;
class Tensor;
namespace runtimes {
namespace opencl {
class OpenCLBufferTransformKernel {
 public:
  virtual MaceStatus Compute(OpContext *context,
                             const Tensor *input,
                             const BufferContentType type,
                             const int wino_blk_size,
                             Tensor *output) = 0;
 MACE_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLBufferTransformKernel)
};
}  // namespace opencl
}  // namespace runtimes

typedef runtimes::opencl::OpenCLBufferTransformKernel
    OpenCLBufferTransformKernel;
}  // namespace mace

#endif  // MACE_RUNTIMES_OPENCL_TRANSFORM_BUFFER_TRANSFORM_KERNEL_H_
