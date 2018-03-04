//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/conv_2d.h"
#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {

extern void Conv2dOpenclK1x1(cl::Kernel *kernel,
                             const Tensor *input,
                             const Tensor *filter,
                             const Tensor *bias,
                             const int stride,
                             const int *padding,
                             const int *dilations,
                             const ActivationType activation,
                             const float relux_max_limit,
                             const float prelu_alpha,
                             const DataType dt,
                             Tensor *output,
                             StatsFuture *future);

extern void Conv2dOpenclK3x3(cl::Kernel *kernel,
                             const Tensor *input,
                             const Tensor *filter,
                             const Tensor *bias,
                             const int stride,
                             const int *padding,
                             const int *dilations,
                             const ActivationType activation,
                             const float relux_max_limit,
                             const float prelu_alpha,
                             const DataType dt,
                             Tensor *output,
                             StatsFuture *future);

extern void Conv2dOpencl(cl::Kernel *kernel,
                         const Tensor *input,
                         const Tensor *filter,
                         const Tensor *bias,
                         const int stride,
                         const int *padding,
                         const int *dilations,
                         const ActivationType activation,
                         const float relux_max_limit,
                         const float prelu_alpha,
                         const DataType dt,
                         Tensor *output,
                         StatsFuture *future);

template<typename T>
void Conv2dFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input,
                                                      const Tensor *filter,
                                                      const Tensor *bias,
                                                      Tensor *output,
                                                      StatsFuture *future) {
  typedef void (*Conv2dOpenclFunction)(
      cl::Kernel *kernel,
      const Tensor *input, const Tensor *filter, const Tensor *bias, const int stride,
      const int *padding, const int *dilations, const ActivationType activation,
      const float relux_max_limit, const float prelu_alpha, const DataType dt,
      Tensor *output, StatsFuture *future);
  // Selection matrix: kernel_size x stride_size
  static const Conv2dOpenclFunction selector[5] =
      {Conv2dOpenclK1x1, nullptr, Conv2dOpenclK3x3, nullptr, nullptr};

  index_t kernel_h = filter->dim(0);
  index_t kernel_w = filter->dim(1);
  if (!input->is_image() || strides_[0] != strides_[1] ||
      (dilations_[0] > 1 && (strides_[0] > 1 || kernel_h == 1))) {
    LOG(WARNING) << "OpenCL conv2d kernel with "
                 << "filter" << kernel_h << "x" << kernel_w << ","
                 << " stride " << strides_[0] << "x" << strides_[1]
                 << ",dilations " << dilations_[0] << "x" << dilations_[1]
                 << " and input image: " << input->is_image()
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
    CalcOutputSize(input->shape().data(), filter->shape().data(), paddings_.data(),
                   dilations_, strides_, RoundType::FLOOR, output_shape.data());
  }

  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, output_image_shape);
  output->ResizeImage(output_shape, output_image_shape);

  if (kernel_h == kernel_w && kernel_h <= 5 &&
      selector[kernel_h - 1] != nullptr) {
    auto conv2d_func = selector[kernel_h - 1];
    conv2d_func(&kernel_, input, filter, bias, strides_[0], paddings.data(), dilations_, activation_,
                relux_max_limit_, prelu_alpha_, DataTypeToEnum<T>::value,
                output, future);
  } else {
    Conv2dOpencl(&kernel_, input, filter, bias, strides_[0], paddings.data(), dilations_,
                 activation_, relux_max_limit_, prelu_alpha_,
                 DataTypeToEnum<T>::value, output, future);
  }
}

template
struct Conv2dFunctor<DeviceType::OPENCL, float>;
template
struct Conv2dFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace
