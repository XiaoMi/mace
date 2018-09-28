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

#include "mace/kernels/split.h"
#include "mace/kernels/opencl/image/split.h"

namespace mace {
namespace kernels {

template <typename T>
SplitFunctor<DeviceType::GPU, T>::SplitFunctor(OpKernelContext *context,
                                               const int32_t axis)
    : OpKernel(context) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::SplitKernel<T>(axis));
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

template <typename T>
MaceStatus SplitFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input,
    const std::vector<Tensor *> &output_list,
    StatsFuture *future) {
  return kernel_->Compute(context_, input, output_list, future);
}

template struct SplitFunctor<DeviceType::GPU, float>;
template struct SplitFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
