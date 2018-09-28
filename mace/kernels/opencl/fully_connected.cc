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

#include "mace/kernels/fully_connected.h"
#include "mace/kernels/opencl/image/fully_connected.h"

namespace mace {
namespace kernels {

template <typename T>
FullyConnectedFunctor<DeviceType::GPU, T>::FullyConnectedFunctor(
    OpKernelContext *context,
    const ActivationType activation,
    const float relux_max_limit)
    : FullyConnectedBase(context, activation, relux_max_limit) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::FullyConnectedKernel<T>);
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}
template <typename T>
MaceStatus FullyConnectedFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input,
    const Tensor *weight,
    const Tensor *bias,
    Tensor *output,
    StatsFuture *future) {
  return kernel_->Compute(
      context_, input, weight, bias, activation_, relux_max_limit_,
      output, future);
}

template struct FullyConnectedFunctor<DeviceType::GPU, float>;

template struct FullyConnectedFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
