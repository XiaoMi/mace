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
#ifndef MACE_OPS_OPENCL_IMAGE_CONV_2D_H_
#define MACE_OPS_OPENCL_IMAGE_CONV_2D_H_

#include "mace/ops/opencl/conv_2d.h"

#include <memory>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

extern MaceStatus Conv2dOpenclK1x1(OpContext *context,
                                   cl::Kernel *kernel,
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
                                   uint32_t *kwg_size);

extern MaceStatus Conv2dOpenclK3x3(OpContext *context,
                                   cl::Kernel *kernel,
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
                                   uint32_t *kwg_size);

extern MaceStatus Conv2dOpencl(OpContext *context,
                               cl::Kernel *kernel,
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
                               uint32_t *kwg_size);


template <typename T>
class Conv2dKernel : public OpenCLConv2dKernel {
 public:
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *bias,
      const int *strides,
      const Padding &padding_type,
      const std::vector<int> &padding_data,
      const int *dilations,
      const ActivationType activation,
      const float relux_max_limit,
      Tensor *output) override;

 private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus Conv2dKernel<T>::Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *bias,
      const int *strides,
      const Padding &padding_type,
      const std::vector<int> &padding_data,
      const int *dilations,
      const ActivationType activation,
      const float relux_max_limit,
      Tensor *output) {
  typedef MaceStatus (*Conv2dOpenclFunction)(
      OpContext *context,
      cl::Kernel *kernel, const Tensor *input, const Tensor *filter,
      const Tensor *bias, const int stride, const int *padding,
      const int *dilations, const ActivationType activation,
      const float relux_max_limit, const DataType dt,
      std::vector<index_t> *input_shape, Tensor *output,
      uint32_t *kwg_size);
  // Selection matrix: kernel_size x stride_size
  static const Conv2dOpenclFunction selector[3] = {
      Conv2dOpenclK1x1, nullptr, Conv2dOpenclK3x3};

  index_t kernel_h = filter->dim(2);
  index_t kernel_w = filter->dim(3);
  if (strides[0] != strides[1] ||
      (dilations[0] > 1 && (strides[0] > 1 || kernel_h == 1))) {
    LOG(WARNING) << "OpenCL conv2d kernel with "
                 << "filter" << kernel_h << "x" << kernel_w << ","
                 << " stride " << strides[0] << "x" << strides[1]
                 << ",dilations " << dilations[0] << "x" << dilations[1]
                 << " is not implemented yet.";
    MACE_NOT_IMPLEMENTED;
  }

  // Reshape output
  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  if (padding_data.empty()) {
    ops::CalcNHWCPaddingAndOutputSize(
        input->shape().data(), filter->shape().data(), dilations, strides,
        padding_type, output_shape.data(), paddings.data());
  } else {
    paddings = padding_data;
    CalcOutputSize(input->shape().data(), filter->shape().data(),
                   padding_data.data(), dilations, strides, RoundType::FLOOR,
                   output_shape.data());
  }

  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL,
                  &output_image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

  if (kernel_h == kernel_w && kernel_h <= 3 &&
      selector[kernel_h - 1] != nullptr) {
    auto conv2d_func = selector[kernel_h - 1];
    return conv2d_func(context,
        &kernel_, input, filter, bias, strides[0], paddings.data(), dilations,
        activation, relux_max_limit, DataTypeToEnum<T>::value, &input_shape_,
        output, &kwg_size_);
  } else {
    return Conv2dOpencl(
        context, &kernel_, input, filter, bias,
        strides[0], paddings.data(), dilations,
        activation, relux_max_limit, DataTypeToEnum<T>::value, &input_shape_,
        output, &kwg_size_);
  }
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_CONV_2D_H_
