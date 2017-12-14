//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/fused_conv_2d.h"
#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {

extern void Conv2dOpenclK1x1S1(const Tensor *input, const Tensor *filter,
                               const Tensor *bias, const bool fused_relu,
                               const int *padding, const DataType dt,
                               Tensor *output,
                               StatsFuture *future);

extern void Conv2dOpenclK1x1S2(const Tensor *input, const Tensor *filter,
                               const Tensor *bias, const bool fused_relu,
                               const int *padding, const DataType dt,
                               Tensor *output,
                               StatsFuture *future);

extern void Conv2dOpenclK3x3S1(const Tensor *input, const Tensor *filter,
                               const Tensor *bias, const bool fused_relu,
                               const int *padding, const DataType dt,
                               Tensor *output,
                               StatsFuture *future);

extern void Conv2dOpenclK3x3S2(const Tensor *input, const Tensor *filter,
                               const Tensor *bias, const bool fused_relu,
                               const int *padding, const DataType dt,
                               Tensor *output,
                               StatsFuture *future);

extern void Conv2dOpencl(const Tensor *input, const Tensor *filter,
                         const Tensor *bias, const bool fused_relu,
                         const uint32_t stride, const int *padding,
                         const DataType dt, Tensor *output,
                         StatsFuture *future);

template<typename T>
void FusedConv2dFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input,
                                                           const Tensor *filter,
                                                           const Tensor *bias,
                                                           Tensor *output,
                                                           StatsFuture *future) {
  typedef void (*Conv2dOpenclFunction)(const Tensor *input, const Tensor *filter,
                                       const Tensor *bias, const bool fused_relu,
                                       const int *padding, const DataType dt,
                                       Tensor *output, StatsFuture *future);
  // Selection matrix: kernel_size x stride_size
  static const Conv2dOpenclFunction selector[5][2] = {
      {Conv2dOpenclK1x1S1, Conv2dOpenclK1x1S2},
      {nullptr, nullptr},
      {Conv2dOpenclK3x3S1, Conv2dOpenclK3x3S2},
      {nullptr, nullptr},
      {nullptr, nullptr}};
  index_t kernel_h = filter->dim(0);
  index_t kernel_w = filter->dim(1);
  if (!input->is_image() || strides_[0] != strides_[1] ||
      strides_[0] > 2 || dilations_[0] != 1 || dilations_[1] != 1) {
    LOG(WARNING) << "OpenCL conv2d kernel with "
                 << "filter" << kernel_h << "x" << kernel_w << ","
                 << " stride " << strides_[0] << "x" << strides_[1]
                 << " is not implemented yet, using slow version";
    MACE_NOT_IMPLEMENTED;
  }

  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  kernels::CalcNHWCPaddingAndOutputSize(
      input->shape().data(), filter->shape().data(), dilations_,
      strides_, paddings_, output_shape.data(), paddings.data());

  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT, output_image_shape);
  output->ResizeImage(output_shape, output_image_shape);

  if (kernel_h == kernel_w && kernel_h <= 5 &&
      selector[kernel_h - 1][strides_[0] - 1] != nullptr) {
    auto conv2d_func = selector[kernel_h - 1][strides_[0] - 1];
    conv2d_func(input, filter, bias, true, paddings.data(),
                DataTypeToEnum<T>::value, output, future);
  } else {
    Conv2dOpencl(input, filter, bias, true, strides_[0], paddings.data(),
                 DataTypeToEnum<T>::value, output, future);
  }
}

template
struct FusedConv2dFunctor<DeviceType::OPENCL, float>;
template
struct FusedConv2dFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace
