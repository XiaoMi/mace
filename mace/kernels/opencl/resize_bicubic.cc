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

#include "mace/kernels/resize_bicubic.h"
#include "mace/kernels/opencl/image/resize_bicubic.h"

namespace mace {
namespace kernels {

template <typename T>
ResizeBicubicFunctor<DeviceType::GPU, T>::ResizeBicubicFunctor(
    OpKernelContext *context,
    bool align_corners,
    const std::vector<index_t> &size)
    : OpKernel(context) {
  MACE_CHECK(size.size() == 2);
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::ResizeBicubicKernel<T>(align_corners,
                                                            size[0],
                                                            size[1]));
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

template <typename T>
MaceStatus ResizeBicubicFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input, Tensor *output, StatsFuture *future) {
  return kernel_->Compute(context_, input, output, future);
}

template struct ResizeBicubicFunctor<DeviceType::GPU, float>;
template struct ResizeBicubicFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
