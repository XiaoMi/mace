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

#include "mace/kernels/activation.h"

#include "mace/kernels/opencl/image/activation.h"

namespace mace {
namespace kernels {

template <typename T>
ActivationFunctor<DeviceType::GPU, T>::ActivationFunctor(
    OpKernelContext *context,
    ActivationType type,
    T relux_max_limit) : OpKernel(context) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(
        new opencl::image::ActivationKernel<T>(type, relux_max_limit));
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}
template <typename T>
MaceStatus ActivationFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input,
    const Tensor *alpha,
    Tensor *output,
    StatsFuture *future) {
  return kernel_->Compute(context_, input, alpha, output, future);
}

template struct ActivationFunctor<DeviceType::GPU, float>;
template struct ActivationFunctor<DeviceType::GPU, half>;
}  // namespace kernels
}  // namespace mace
