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

#include "mace/kernels/buffer_transform.h"
#include "mace/kernels/opencl/image/buffer_to_image.h"
#include "mace/kernels/opencl/buffer/buffer_transform.h"

namespace mace {
namespace kernels {

template<typename T>
BufferTransformFunctor<DeviceType::GPU, T>::BufferTransformFunctor(
    OpKernelContext *context,
    const int wino_blk_size)
  : BufferTransformFunctorBase(context, wino_blk_size) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::BufferToImage<T>);
  } else {
    kernel_.reset(new opencl::buffer::BufferTransform<T>);
  }
}

template <typename T>
MaceStatus BufferTransformFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input,
    const BufferType type,
    Tensor *output,
    StatsFuture *future) {
  return kernel_->Compute(context_, input, type,
                          wino_blk_size_, output, future);
}

template struct BufferTransformFunctor<DeviceType::GPU, float>;
template struct BufferTransformFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
