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

#include "mace/kernels/eltwise.h"
#include "mace/kernels/opencl/image/eltwise.h"

namespace mace {
namespace kernels {
template <typename T>
EltwiseFunctor<DeviceType::GPU, T>::EltwiseFunctor(
    OpKernelContext *context,
    const EltwiseType type,
    const std::vector<float> &coeff,
    const float scalar_input,
    const int32_t scalar_input_index,
    const DataFormat data_format) : OpKernel(context) {
  MACE_UNUSED(data_format);
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::EltwiseKernel<T>(
        type, coeff, scalar_input, scalar_input_index));
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

template <typename T>
MaceStatus EltwiseFunctor<DeviceType::GPU, T>::operator()(const Tensor *input0,
                                                          const Tensor *input1,
                                                          Tensor *output,
                                                          StatsFuture *future) {
  return kernel_->Compute(context_, input0, input1, output, future);
}

template struct EltwiseFunctor<DeviceType::GPU, float>;
template struct EltwiseFunctor<DeviceType::GPU, half>;
}  // namespace kernels
}  // namespace mace
