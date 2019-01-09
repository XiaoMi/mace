// Copyright 2018 The MACE Authors. All Rights Reserved.
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
#ifndef MACE_OPS_OPENCL_IMAGE_DEPTHWISE_CONV2D_H_
#define MACE_OPS_OPENCL_IMAGE_DEPTHWISE_CONV2D_H_

#include "mace/ops/opencl/depthwise_conv2d.h"

#include <memory>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {
namespace depthwise {

MaceStatus DepthwiseConv2d(OpContext *context,
                           cl::Kernel *kernel,
                           const Tensor *input,   // NHWC
                           const Tensor *filter,  // HWIM
                           const Tensor *bias,
                           const int stride,
                           const int *paddings,
                           const int *dilations,
                           const ActivationType activation,
                           const float relux_max_limit,
                           const float leakyrelu_coefficient,
                           const DataType dt,
                           std::vector<index_t> *prev_input_shape,
                           Tensor *output,
                           uint32_t *kwg_size);
}  // namespace depthwise


template <typename T>
class DepthwiseConv2dKernel : public OpenCLDepthwiseConv2dKernel {
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
      const float leakyrelu_coefficient,
      Tensor *output) override;

 private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus DepthwiseConv2dKernel<T>::Compute(
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
    const float leakyrelu_coefficient,
    Tensor *output) {
  index_t kernel_h = filter->dim(2);
  index_t kernel_w = filter->dim(3);
  if (strides[0] != strides[1]) {
    LOG(WARNING) << "OpenCL depthwise conv2d kernel with "
                 << "filter" << kernel_h << "x" << kernel_w << ","
                 << " stride " << strides[0] << "x" << strides[1]
                 << " is not implemented yet, using slow version";
    MACE_NOT_IMPLEMENTED;
  }

  // Create a fake conv_2d filter to calculate the paddings and output size
  std::vector<index_t> fake_filter_shape(4);
  fake_filter_shape[0] = filter->dim(0) * filter->dim(1);
  fake_filter_shape[1] = filter->dim(1);
  fake_filter_shape[2] = filter->dim(2);
  fake_filter_shape[3] = filter->dim(3);

  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  if (padding_data.empty()) {
    ops::CalcNHWCPaddingAndOutputSize(
        input->shape().data(), fake_filter_shape.data(), dilations, strides,
        padding_type, output_shape.data(), paddings.data());
  } else {
    paddings = padding_data;
    CalcOutputSize(input->shape().data(), fake_filter_shape.data(),
                   padding_data.data(), dilations, strides, RoundType::FLOOR,
                   output_shape.data());
  }

  std::vector<size_t> output_image_shape;
  OpenCLUtil::CalImage2DShape(output_shape, OpenCLBufferType::IN_OUT_CHANNEL,
                              &output_image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

  return depthwise::DepthwiseConv2d(
      context, &kernel_, input, filter, bias, strides[0], paddings.data(),
      dilations, activation, relux_max_limit, leakyrelu_coefficient,
      DataTypeToEnum<T>::value, &input_shape_, output, &kwg_size_);
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_DEPTHWISE_CONV2D_H_
