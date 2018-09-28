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

#include "mace/kernels/deconv_2d.h"
#include "mace/kernels/opencl/image/deconv_2d.h"

namespace mace {
namespace kernels {

template <typename T>
Deconv2dFunctor<DeviceType::GPU, T>::Deconv2dFunctor(
    OpKernelContext *context,
    const std::vector<int> &strides,
    const Padding &padding_type,
    const std::vector<int> &paddings,
    const std::vector<index_t> &output_shape,
    const ActivationType activation,
    const float relux_max_limit)
    : Deconv2dFunctorBase(context,
                          strides,
                          padding_type,
                          paddings,
                          output_shape,
                          activation,
                          relux_max_limit) {
  if (context->device()->opencl_runtime()->UseImageMemory()) {
    kernel_.reset(new opencl::image::Deconv2dKernel<T>);
  } else {
    MACE_NOT_IMPLEMENTED;
  }
}

template <typename T>
MaceStatus Deconv2dFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input,
    const Tensor *filter,
    const Tensor *bias,
    const Tensor *output_shape_tensor,
    Tensor *output,
    StatsFuture *future) {
  MACE_CHECK_NOTNULL(input);
  MACE_CHECK_NOTNULL(filter);
  MACE_CHECK_NOTNULL(output);
  std::vector<int> paddings(2);
  std::vector<index_t> output_shape(4);
  if (paddings_.empty()) {
    paddings = std::vector<int>(2, 0);
    if (output_shape_.size() != 4) {
      MACE_CHECK_NOTNULL(output_shape_tensor);
      MACE_CHECK(output_shape_tensor->size() == 4);
      Tensor::MappingGuard output_shape_mapper(output_shape_tensor);
      auto output_shape_data =
          output_shape_tensor->data<int32_t>();
      output_shape =
          std::vector<index_t>(output_shape_data, output_shape_data + 4);
    } else {
      output_shape = output_shape_;
    }
    CalcDeconvPaddingAndInputSize(input->shape().data(),
                                  filter->shape().data(),
                                  strides_.data(),
                                  padding_type_,
                                  output_shape.data(),
                                  paddings.data());
  } else {
    paddings = paddings_;
    output_shape = std::vector<index_t>(4, 0);
    CalcDeconvOutputSize(input->shape().data(),
                         filter->shape().data(),
                         strides_.data(),
                         output_shape.data(),
                         paddings.data());
  }

  return kernel_->Compute(context_, input, filter, bias,
                          strides_.data(), paddings.data(), activation_,
                          relux_max_limit_, output_shape, output, future);
}

template struct Deconv2dFunctor<DeviceType::GPU, float>;
template struct Deconv2dFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
