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

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {

struct BufferToImageFunctorBase {
  explicit BufferToImageFunctorBase(bool i2b)
    : i2b_(i2b), kernel_error_(nullptr) {}
  bool i2b_;
  std::unique_ptr<BufferBase> kernel_error_;
};

template <DeviceType D, typename T>
struct BufferToImageFunctor : BufferToImageFunctorBase {
  explicit BufferToImageFunctor(bool i2b = false)
      : BufferToImageFunctorBase(i2b) {}
  void operator()(Tensor *input,
                  const BufferType type,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_NOT_IMPLEMENTED;
  }
};

template <typename T>
struct BufferToImageFunctor<DeviceType::GPU, T> : BufferToImageFunctorBase {
  explicit BufferToImageFunctor(bool i2b = false)
      : BufferToImageFunctorBase(i2b) {}
  void operator()(Tensor *input,
                  const BufferType type,
                  Tensor *output,
                  StatsFuture *future);
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_BUFFER_TO_IMAGE_H_
