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

#ifndef MACE_KERNELS_OPENCL_BATCH_TO_SPACE_H_
#define MACE_KERNELS_OPENCL_BATCH_TO_SPACE_H_

#include "mace/kernels/batch_to_space.h"
#include "mace/kernels/opencl/image/batch_to_space.h"

namespace mace {
namespace kernels {

template <typename T>
BatchToSpaceFunctor<DeviceType::GPU, T>::BatchToSpaceFunctor(
    OpKernelContext *context,
    const std::vector<int> &paddings,
    const std::vector<int> &block_shape)
    : BatchToSpaceFunctorBase(context, paddings, block_shape) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::BatchToSpaceKernel<T>);
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}
template <typename T>
MaceStatus BatchToSpaceFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *batch_tensor, Tensor *space_tensor, StatsFuture *future) {
  std::vector<index_t> output_shape(4, 0);
  CalculateBatchToSpaceOutputShape(batch_tensor, DataFormat::NHWC,
                                   output_shape.data());
  return kernel_->Compute(context_, batch_tensor, paddings_, block_shape_,
                          output_shape, space_tensor, future);
}

template struct BatchToSpaceFunctor<DeviceType::GPU, float>;
template struct BatchToSpaceFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
#endif  // MACE_KERNELS_OPENCL_BATCH_TO_SPACE_H_
