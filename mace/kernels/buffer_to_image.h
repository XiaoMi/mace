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

#ifndef MACE_KERNELS_BUFFER_TO_IMAGE_H_
#define MACE_KERNELS_BUFFER_TO_IMAGE_H_

#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/opencl/common.h"

namespace mace {
namespace kernels {

struct BufferToImageFunctorBase {
  explicit BufferToImageFunctorBase(const int wino_blk_size)
    : wino_blk_size_(wino_blk_size) {}
  const int wino_blk_size_;
};

template <DeviceType D, typename T>
struct BufferToImageFunctor : BufferToImageFunctorBase {
  explicit BufferToImageFunctor(const int wino_blk_size)
    : BufferToImageFunctorBase(wino_blk_size) {}
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

template <typename T>
struct BufferToImageFunctor<DeviceType::GPU, T> : BufferToImageFunctorBase {
  explicit BufferToImageFunctor(const int wino_blk_size)
      : BufferToImageFunctorBase(wino_blk_size) {}
  MaceStatus operator()(const Tensor *input,
                  const BufferType type,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_BUFFER_TO_IMAGE_H_
