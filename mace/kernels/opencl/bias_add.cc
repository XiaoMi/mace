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

#include "mace/kernels/bias_add.h"
#include "mace/kernels/opencl/image/bias_add.h"

namespace mace {
namespace kernels {

template <typename T>
BiasAddFunctor<DeviceType::GPU, T>::BiasAddFunctor(
    OpKernelContext *context,
    const DataFormat data_format)
    : BiasAddFunctorBase(context, data_format) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::BiasAddKernel<T>);
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

template <typename T>
MaceStatus BiasAddFunctor<DeviceType::GPU, T>::operator()(const Tensor *input,
                                                          const Tensor *bias,
                                                          Tensor *output,
                                                          StatsFuture *future) {
  MACE_CHECK(input->dim_size() == 4 && data_format_ == NHWC,
             "gpu only support biasadd for 4-dimensional NHWC format tensor");
  return kernel_->Compute(context_, input, bias, output, future);
}

template struct BiasAddFunctor<DeviceType::GPU, float>;
template struct BiasAddFunctor<DeviceType::GPU, half>;
}  // namespace kernels
}  // namespace mace
