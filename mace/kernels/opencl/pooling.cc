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

#include "mace/kernels/pooling.h"

#include "mace/kernels/opencl/buffer/pooling.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/kernels/opencl/image/pooling.h"

namespace mace {
namespace kernels {

template <typename T>
PoolingFunctor<DeviceType::GPU, T>::PoolingFunctor(
    OpKernelContext *context,
    const PoolingType pooling_type,
    const int *kernels,
    const int *strides,
    const Padding padding_type,
    const std::vector<int> &paddings,
    const int *dilations)
    : PoolingFunctorBase(context,
                         pooling_type,
                         kernels,
                         strides,
                         padding_type,
                         paddings,
                         dilations) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::PoolingKernel<T>);
  } else {
    kernel_.reset(new opencl::buffer::PoolingKernel<T>);
  }
}

template <typename T>
MaceStatus PoolingFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input,
    Tensor *output,
    StatsFuture *future) {
  return kernel_->Compute(context_, input, pooling_type_, kernels_, strides_,
                          padding_type_, paddings_, dilations_,
                          output, future);
}

template struct PoolingFunctor<DeviceType::GPU, float>;
template struct PoolingFunctor<DeviceType::GPU, half>;
}  // namespace kernels
}  // namespace mace
