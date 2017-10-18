//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/conv_2d.h"
#include "mace/kernels/conv_pool_2d_util.h"

namespace mace {
namespace kernels {

extern void Conv2dOpenclK1x1S1(const Tensor *input, const Tensor *filter,
                               const Tensor *bias, Tensor *output);

template <>
void Conv2dFunctor<DeviceType::OPENCL, float>::operator()(const Tensor *input,
                                                          const Tensor *filter,
                                                          const Tensor *bias,
                                                          Tensor *output) {
  typedef void (*Conv2dOpenclFunction)(const Tensor *input, const Tensor *filter,
                                       const Tensor *bias, Tensor *output);
  // Selection matrix: kernel_size x stride_size
  static const Conv2dOpenclFunction selector[5][2] = {
      {Conv2dOpenclK1x1S1, nullptr},
      {nullptr, nullptr},
      {nullptr, nullptr},
      {nullptr, nullptr},
      {nullptr, nullptr}};

  index_t kernel_h = filter->shape()[2];
  index_t kernel_w = filter->shape()[3];
  if (kernel_h != kernel_w || kernel_h > 5 || strides_[0] != strides_[1] ||
      strides_[0] > 2 || dilations_[0] != 1 || dilations_[1] != 1 ||
      selector[kernel_h - 1][strides_[0] - 1] == nullptr) {
    LOG(WARNING) << "OpenCL conv2d kernel with "
                 << "filter" << kernel_h << "x" << kernel_w << ","
                 << " stride " << strides_[0] << "x" << strides_[1]
                 << " is not implemented yet, using slow version";
    // TODO(heliangliang) The CPU/NEON kernel should map the buffer
    Conv2dFunctor<DeviceType::CPU, float>(strides_, paddings_, dilations_)(
        input, filter, bias, output);
    return;
  }

  MACE_CHECK(paddings_[0] == 1 && paddings_[1] == 1, "Padding not supported");

  auto conv2d_func = selector[kernel_h - 1][strides_[0] - 1];
  conv2d_func(input, filter, bias, output);
}

}  // namespace kernels
}  // namespace mace
