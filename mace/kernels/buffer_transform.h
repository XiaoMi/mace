// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#ifndef MACE_KERNELS_BUFFER_TRANSFORM_H_
#define MACE_KERNELS_BUFFER_TRANSFORM_H_

#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/kernel.h"
#include "mace/kernels/opencl/common.h"

namespace mace {
namespace kernels {

struct BufferTransformFunctorBase : OpKernel {
  explicit BufferTransformFunctorBase(OpKernelContext *context,
                                      const int wino_blk_size)
    : OpKernel(context), wino_blk_size_(wino_blk_size) {}
  const int wino_blk_size_;
};

template <DeviceType D, typename T>
struct BufferTransformFunctor : BufferTransformFunctorBase {
  BufferTransformFunctor(OpKernelContext *context,
                         const int wino_blk_size)
      : BufferTransformFunctorBase(context, wino_blk_size) {}

  MaceStatus operator()(const Tensor *input,
                        const BufferType type,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(input);
    MACE_UNUSED(type);
    MACE_UNUSED(output);
    MACE_UNUSED(future);
    MACE_NOT_IMPLEMENTED;
    return MACE_SUCCESS;
  }
};

class OpenCLBufferTransformKernel {
 public:
  virtual MaceStatus Compute(OpKernelContext *context,
                             const Tensor *input,
                             const BufferType type,
                             const int wino_blk_size,
                             Tensor *output,
                             StatsFuture *future) = 0;
  MACE_VIRTUAL_EMPTY_DESTRUCTOR(OpenCLBufferTransformKernel)
};

template <typename T>
struct BufferTransformFunctor<DeviceType::GPU, T> : BufferTransformFunctorBase {
  BufferTransformFunctor(OpKernelContext *context, const int wino_blk_size);

  MaceStatus operator()(const Tensor *input,
                        const BufferType type,
                        Tensor *output,
                        StatsFuture *future);

  std::unique_ptr<OpenCLBufferTransformKernel> kernel_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_BUFFER_TRANSFORM_H_
