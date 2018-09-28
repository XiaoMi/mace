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

#include "mace/kernels/softmax.h"

#include "mace/kernels/opencl/buffer/softmax.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/kernels/opencl/image/softmax.h"

namespace mace {
namespace kernels {

template <typename T>
SoftmaxFunctor<DeviceType::GPU, T>::SoftmaxFunctor(OpKernelContext *context)
    : OpKernel(context) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::SoftmaxKernel<T>);
  } else {
    kernel_.reset(new opencl::buffer::SoftmaxKernel<T>);
  }
}
template <typename T>
MaceStatus SoftmaxFunctor<DeviceType::GPU, T>::operator()(const Tensor *logits,
                                                          Tensor *output,
                                                          StatsFuture *future) {
  return kernel_->Compute(context_, logits, output, future);
}

template struct SoftmaxFunctor<DeviceType::GPU, float>;
template struct SoftmaxFunctor<DeviceType::GPU, half>;
}  // namespace kernels
}  // namespace mace
