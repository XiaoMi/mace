//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/conv_2d.h"
#include "mace/kernels/depthwise_conv2d.h"

namespace mace {
namespace kernels {

extern void Conv2dNeonK3x3S1(const float *input,
                             const index_t *input_shape,
                             const float *filter,
                             const index_t *filter_shape,
                             const float *bias,
                             float *output,
                             const index_t *output_shape);

extern void Conv2dNeonK3x3S2(const float *input,
                             const index_t *input_shape,
                             const float *filter,
                             const index_t *filter_shape,
                             const float *bias,
                             float *output,
                             const index_t *output_shape);

template <>
void DepthwiseConv2dFunctor<DeviceType::NEON, float>::operator()(
    const Tensor *input,
    const Tensor *filter,
    const Tensor *bias,
    Tensor *output) {
  typedef void (*Conv2dNeonFunction)(
      const float *input, const index_t *input_shape, const float *filter,
      const index_t *filter_shape, const float *bias, float *output,
      const index_t *output_shape);
  // Selection matrix: kernel_size x stride_size
  static const Conv2dNeonFunction selector[5][2] = {
      {nullptr, nullptr},
      {nullptr, nullptr},
      {Conv2dNeonK3x3S1, Conv2dNeonK3x3S2},
      {nullptr, nullptr},
      {nullptr, nullptr}};
  // not implement yet
  index_t kernel_h = filter->dim(2);
  index_t kernel_w = filter->dim(3);
  if (kernel_h != kernel_w || kernel_h > 5 || strides_[0] != strides_[1] ||
      strides_[0] > 2 || dilations_[0] != 1 || dilations_[1] != 1 ||
      selector[kernel_h - 1][strides_[0] - 1] == nullptr) {
    LOG(WARNING) << "Depthwise-Conv2d NEON kernel with "
                 << "filter" << kernel_h << "x" << kernel_w << ","
                 << " stride " << strides_[0] << "x" << strides_[1]
                 << " is not implemented yet, using slow version";
    DepthwiseConv2dFunctor<DeviceType::CPU, float>(strides_, paddings_,
                                                   dilations_)(
        input, filter, bias, output);
    return;
  }

  const float *input_ptr = input->data<float>();
  const index_t *input_shape = input->shape().data();
  const float *filter_ptr = filter->data<float>();
  const index_t *filter_shape = filter->shape().data();
  const float *bias_ptr = bias->data<float>();
  float *output_ptr = output->mutable_data<float>();
  const index_t *output_shape = output->shape().data();
  // Keep this alive during kernel execution
  Tensor padded_input;
  if (paddings_[0] > 0 || paddings_[1] > 0) {
    ConstructInputWithPadding(input, paddings_.data(), &padded_input);
    input_ptr = padded_input.data<float>();
    input_shape = padded_input.shape().data();
  }
  auto conv2d_neon_func = selector[kernel_h - 1][strides_[0] - 1];
  conv2d_neon_func(input_ptr, input_shape, filter_ptr, filter_shape, bias_ptr, output_ptr,
                   output_shape);
}

}  //  namespace kernels
}  //  namespace mace