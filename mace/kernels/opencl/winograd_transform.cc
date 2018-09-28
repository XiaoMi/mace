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

#include "mace/kernels/winograd_transform.h"
#include "mace/kernels/opencl/image/winograd_transform.h"

namespace mace {
namespace kernels {

template <typename T>
WinogradTransformFunctor<DeviceType::GPU, T>::WinogradTransformFunctor(
    OpKernelContext *context,
    const Padding &padding_type,
    const std::vector<int> &paddings,
    const int block_size) : OpKernel(context) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::WinogradTransformKernel<T>(
        padding_type, paddings, block_size));
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}
template <typename T>
MaceStatus WinogradTransformFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input_tensor, Tensor *output_tensor, StatsFuture *future) {
  return kernel_->Compute(context_, input_tensor, output_tensor, future);
}

template <typename T>
WinogradInverseTransformFunctor<DeviceType::GPU, T>::WinogradInverseTransformFunctor(  // NOLINT(whitespace/line_length)
    OpKernelContext *context,
    const ActivationType activation,
    const float relux_max_limit,
    const int block_size) : OpKernel(context) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::WinogradInverseTransformKernel<T>(
        activation, relux_max_limit, block_size));
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}
template <typename T>
MaceStatus WinogradInverseTransformFunctor<DeviceType::GPU, T>::operator()(
    const std::vector<const Tensor*> &inputs,
    Tensor *output_tensor,
    StatsFuture *future) {
  return kernel_->Compute(context_, inputs, output_tensor, future);
}

template struct WinogradTransformFunctor<DeviceType::GPU, float>;
template struct WinogradTransformFunctor<DeviceType::GPU, half>;

template struct WinogradInverseTransformFunctor<DeviceType::GPU, float>;
template struct WinogradInverseTransformFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
