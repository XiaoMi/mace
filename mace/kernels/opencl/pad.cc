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

#include "mace/kernels/pad.h"
#include "mace/kernels/opencl/image/pad.h"

namespace mace {
namespace kernels {

template <typename T>
PadFunctor<DeviceType::GPU, T>::PadFunctor(
    OpKernelContext *context,
    const std::vector<int> &paddings,
    const float constant_value)
    : OpKernel(context) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::PadKernel<T>(paddings, constant_value));
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

template <typename T>
MaceStatus PadFunctor<DeviceType::GPU, T>::operator()(const Tensor *input,
                                                      Tensor *output,
                                                      StatsFuture *future) {
  return kernel_->Compute(context_, input, output, future);
}

template struct PadFunctor<DeviceType::GPU, float>;
template struct PadFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
