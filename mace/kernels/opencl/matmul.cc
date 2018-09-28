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

#include "mace/kernels/matmul.h"
#include "mace/kernels/opencl/image/matmul.h"

namespace mace {
namespace kernels {

template <typename T>
MatMulFunctor<DeviceType::GPU, T>::MatMulFunctor(OpKernelContext *context)
    : OpKernel(context) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::MatMulKernel<T>);
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

template <typename T>
MaceStatus MatMulFunctor<DeviceType::GPU, T>::operator()(const Tensor *A,
                                                         const Tensor *B,
                                                         Tensor *C,
                                                         bool transpose_a,
                                                         bool transpose_b,
                                                         StatsFuture *future) {
  return kernel_->Compute(context_, A, B, C, transpose_a, transpose_b, future);
}

template struct MatMulFunctor<DeviceType::GPU, float>;

template struct MatMulFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
