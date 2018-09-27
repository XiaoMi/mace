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

#include "mace/kernels/sqrdiff_mean.h"
#include "mace/kernels/opencl/image/sqrdiff_mean.h"

namespace mace {
namespace kernels {

template <typename T>
SqrDiffMeanFunctor<DeviceType::GPU, T>::SqrDiffMeanFunctor(
    OpKernelContext *context) : OpKernel(context) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::SqrDiffMeanKernel<T>());
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

template <typename T>
MaceStatus SqrDiffMeanFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input0,
    const Tensor *input1,
    Tensor *output,
    StatsFuture *future) {
  return kernel_->Compute(context_, input0, input1, output, future);
}

template struct SqrDiffMeanFunctor<DeviceType::GPU, float>;
template struct SqrDiffMeanFunctor<DeviceType::GPU, half>;
}  // namespace kernels
}  // namespace mace
