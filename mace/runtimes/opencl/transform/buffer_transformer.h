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

#ifndef MACE_RUNTIMES_OPENCL_TRANSFORM_BUFFER_TRANSFORMER_H_
#define MACE_RUNTIMES_OPENCL_TRANSFORM_BUFFER_TRANSFORMER_H_

#include "mace/core/ops/operator.h"
#include "mace/runtimes/opencl/transform/buffer_transform.h"

namespace mace {
namespace runtimes {
namespace opencl {
// Only used for GPU Operation(BufferTransform)
class OpenCLBufferTransformer {
 public:
  OpenCLBufferTransformer(const MemoryType in_mem_type,
                          const MemoryType out_mem_type);

  MaceStatus Transform(OpContext *context,
                       const Tensor *input,
                       const BufferContentType type,
                       const MemoryType out_mem_type,
                       const int wino_blk_size,
                       Tensor *output);

 private:
  std::unique_ptr<OpenCLBufferTransformKernel> kernel_;
};

}  // namespace opencl
}  // namespace runtimes

typedef runtimes::opencl::OpenCLBufferTransformer OpenCLBufferTransformer;

MaceStatus TransformFilter(
    mace::OpConstructContext *context,
    OperatorDef *op_def,
    const int input_idx,
    const BufferContentType buffer_type,
    const MemoryType mem_type,
    const int wino_blk_size = 0);

}  // namespace mace

#endif  // MACE_RUNTIMES_OPENCL_TRANSFORM_BUFFER_TRANSFORMER_H_
