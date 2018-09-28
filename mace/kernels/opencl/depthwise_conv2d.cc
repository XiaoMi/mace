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

#include "mace/kernels/depthwise_conv2d.h"
#include "mace/kernels/opencl/buffer/depthwise_conv2d.h"
#include "mace/kernels/opencl/image/depthwise_conv2d.h"

namespace mace {
namespace kernels {
template <typename T>
DepthwiseConv2dFunctor<DeviceType::GPU, T>::DepthwiseConv2dFunctor(
    OpKernelContext *context,
    const int *strides,
    const Padding padding_type,
    const std::vector<int> &paddings,
    const int *dilations,
    const ActivationType activation,
    const float relux_max_limit)
    : DepthwiseConv2dFunctorBase(context,
                                 strides,
                                 padding_type,
                                 paddings,
                                 dilations,
                                 activation,
                                 relux_max_limit) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::DepthwiseConv2dKernel<T>);
  } else {
    kernel_.reset(new opencl::buffer::DepthwiseConv2dKernel<T>);
  }
}

template <typename T>
MaceStatus DepthwiseConv2dFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input,
    const Tensor *filter, /* MIHW */
    const Tensor *bias,
    Tensor *output,
    StatsFuture *future) {
  return kernel_->Compute(context_, input, filter, bias,
                          strides_, padding_type_, paddings_,
                          dilations_, activation_, relux_max_limit_,
                          output, future);
}

template struct DepthwiseConv2dFunctor<DeviceType::GPU, float>;
template struct DepthwiseConv2dFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
