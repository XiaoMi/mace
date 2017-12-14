//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/conv_2d.h"
#include "mace/kernels/conv_pool_2d_util.h"

namespace mace {
namespace kernels {

extern void Conv2dNeonK1x1S1(const float *input,
                             const index_t *input_shape,
                             const float *filter,
                             const index_t *filter_shape,
                             const float *bias,
                             float *output,
                             const index_t *output_shape);

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

extern void Conv2dNeonK5x5S1(const float *input,
                             const index_t *input_shape,
                             const float *filter,
                             const index_t *filter_shape,
                             const float *bias,
                             float *output,
                             const index_t *output_shape);

template <>
void Conv2dFunctor<DeviceType::NEON, float>::operator()(const Tensor *input,
                                                        const Tensor *filter,
                                                        const Tensor *bias,
                                                        Tensor *output,
                                                        StatsFuture *future) {
  MACE_CHECK_NOTNULL(input);
  MACE_CHECK_NOTNULL(filter);
  MACE_CHECK_NOTNULL(output);


  std::vector<index_t> output_shape_vec(4);
  std::vector<int> paddings(2);
  kernels::CalcPaddingAndOutputSize(
      input->shape().data(), filter->shape().data(), dilations_,
      strides_, paddings_, output_shape_vec.data(), paddings.data());
  output->Resize(output_shape_vec);

  typedef void (*Conv2dNeonFunction)(
      const float *input, const index_t *input_shape, const float *filter,
      const index_t *filter_shape, const float *bias, float *output,
      const index_t *output_shape);
  // Selection matrix: kernel_size x stride_size
  static const Conv2dNeonFunction selector[5][2] = {
      {Conv2dNeonK1x1S1, nullptr},
      {nullptr, nullptr},
      {Conv2dNeonK3x3S1, Conv2dNeonK3x3S2},
      {nullptr, nullptr},
      {Conv2dNeonK5x5S1, nullptr}};
  // not implement yet
  index_t kernel_h = filter->dim(2);
  index_t kernel_w = filter->dim(3);
  if (kernel_h != kernel_w || kernel_h > 5 || strides_[0] != strides_[1] ||
      strides_[0] > 2 || dilations_[0] != 1 || dilations_[1] != 1 ||
      selector[kernel_h - 1][strides_[0] - 1] == nullptr) {
    LOG(WARNING) << "NEON conv2d kernel with "
                 << "filter" << kernel_h << "x" << kernel_w << ","
                 << " stride " << strides_[0] << "x" << strides_[1]
                 << " is not implemented yet, using slow version";
    Conv2dFunctor<DeviceType::CPU, float>(strides_, paddings_, dilations_)(
        input, filter, bias, output, future);
    return;
  }

  Tensor padded_input;
  // Keep this alive during kernel execution
  if (paddings[0] > 0 || paddings[1] > 0) {
    ConstructInputWithPadding(input, paddings.data(), &padded_input);
    input = &padded_input;
  }
  Tensor::MappingGuard input_mapper(input);
  Tensor::MappingGuard filter_mapper(filter);
  Tensor::MappingGuard bias_mapper(bias);
  Tensor::MappingGuard output_mapper(output);
  auto input_data = input->data<float>();
  auto input_shape = input->shape().data();
  auto filter_data = filter->data<float>();
  auto bias_data = bias == nullptr ? nullptr : bias->data<float>();
  auto output_data = output->mutable_data<float>();
  auto output_shape = output->shape().data();

  auto conv2d_neon_func = selector[kernel_h - 1][strides_[0] - 1];
  conv2d_neon_func(input_data, input_shape, filter_data, nullptr,
                   bias_data, output_data, output_shape);
}

}  //  namespace kernels
}  //  namespace mace
