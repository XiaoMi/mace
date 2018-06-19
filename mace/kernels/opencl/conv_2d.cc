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

#include "mace/kernels/conv_2d.h"
#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {

extern MaceStatus Conv2dOpenclK1x1(cl::Kernel *kernel,
                                   const Tensor *input,
                                   const Tensor *filter,
                                   const Tensor *bias,
                                   const int stride,
                                   const int *padding,
                                   const int *dilations,
                                   const ActivationType activation,
                                   const float relux_max_limit,
                                   const DataType dt,
                                   std::vector<index_t> *prev_input_shape,
                                   Tensor *output,
                                   StatsFuture *future,
                                   uint32_t *kwg_size,
                                   std::unique_ptr<BufferBase> *kernel_error);

extern MaceStatus Conv2dOpenclK3x3(cl::Kernel *kernel,
                                   const Tensor *input,
                                   const Tensor *filter,
                                   const Tensor *bias,
                                   const int stride,
                                   const int *padding,
                                   const int *dilations,
                                   const ActivationType activation,
                                   const float relux_max_limit,
                                   const DataType dt,
                                   std::vector<index_t> *prev_input_shape,
                                   Tensor *output,
                                   StatsFuture *future,
                                   uint32_t *kwg_size,
                                   std::unique_ptr<BufferBase> *kernel_error);

extern MaceStatus Conv2dOpencl(cl::Kernel *kernel,
                               const Tensor *input,
                               const Tensor *filter,
                               const Tensor *bias,
                               const int stride,
                               const int *padding,
                               const int *dilations,
                               const ActivationType activation,
                               const float relux_max_limit,
                               const DataType dt,
                               std::vector<index_t> *prev_input_shape,
                               Tensor *output,
                               StatsFuture *future,
                               uint32_t *kwg_size,
                               std::unique_ptr<BufferBase> *kernel_error);

template <typename T>
MaceStatus Conv2dFunctor<DeviceType::GPU, T>::operator()(const Tensor *input,
                                                         const Tensor *filter,
                                                         const Tensor *bias,
                                                         Tensor *output,
                                                         StatsFuture *future) {
  typedef MaceStatus (*Conv2dOpenclFunction)(
      cl::Kernel * kernel, const Tensor *input, const Tensor *filter,
      const Tensor *bias, const int stride, const int *padding,
      const int *dilations, const ActivationType activation,
      const float relux_max_limit, const DataType dt,
      std::vector<index_t> *input_shape, Tensor *output, StatsFuture *future,
      uint32_t *kwg_size, std::unique_ptr<BufferBase> *kernel_error);
  // Selection matrix: kernel_size x stride_size
  static const Conv2dOpenclFunction selector[3] = {
      Conv2dOpenclK1x1, nullptr, Conv2dOpenclK3x3};

  index_t kernel_h = filter->dim(2);
  index_t kernel_w = filter->dim(3);
  if (strides_[0] != strides_[1] ||
      (dilations_[0] > 1 && (strides_[0] > 1 || kernel_h == 1))) {
    LOG(WARNING) << "OpenCL conv2d kernel with "
                 << "filter" << kernel_h << "x" << kernel_w << ","
                 << " stride " << strides_[0] << "x" << strides_[1]
                 << ",dilations " << dilations_[0] << "x" << dilations_[1]
                 << " is not implemented yet.";
    MACE_NOT_IMPLEMENTED;
  }

  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  if (paddings_.empty()) {
    kernels::CalcNHWCPaddingAndOutputSize(
        input->shape().data(), filter->shape().data(), dilations_, strides_,
        padding_type_, output_shape.data(), paddings.data());
  } else {
    paddings = paddings_;
    CalcOutputSize(input->shape().data(), filter->shape().data(),
                   paddings_.data(), dilations_, strides_, RoundType::FLOOR,
                   output_shape.data());
  }

  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL,
                  &output_image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

  if (kernel_h == kernel_w && kernel_h <= 3 &&
      selector[kernel_h - 1] != nullptr) {
    auto conv2d_func = selector[kernel_h - 1];
    return conv2d_func(
        &kernel_, input, filter, bias, strides_[0], paddings.data(), dilations_,
        activation_, relux_max_limit_, DataTypeToEnum<T>::value, &input_shape_,
        output, future, &kwg_size_, &kernel_error_);
  } else {
    return Conv2dOpencl(
        &kernel_, input, filter, bias, strides_[0], paddings.data(), dilations_,
        activation_, relux_max_limit_, DataTypeToEnum<T>::value, &input_shape_,
        output, future, &kwg_size_, &kernel_error_);
  }
}

template struct Conv2dFunctor<DeviceType::GPU, float>;
template struct Conv2dFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
