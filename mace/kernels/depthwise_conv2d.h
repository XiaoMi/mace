//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_DEPTHWISE_CONV2D_H_
#define MACE_KERNELS_DEPTHWISE_CONV2D_H_

#include "mace/core/common.h"
#include "mace/core/future.h"
#include "mace/core/public/mace.h"
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/core/runtime/opencl/cl2_header.h"

namespace mace {
namespace kernels {

struct DepthwiseConv2dFunctorBase {
  DepthwiseConv2dFunctorBase(const int *strides,
                             const Padding padding,
                             const int *dilations,
                             const ActivationType activation,
                             const float relux_max_limit,
                             const float prelu_alpha)
      : strides_(strides),
        padding_(padding),
        dilations_(dilations),
        activation_(activation),
        relux_max_limit_(relux_max_limit),
        prelu_alpha_(prelu_alpha) {}

  const int *strides_;    // [stride_h, stride_w]
  const Padding padding_;
  const int *dilations_;  // [dilation_h, dilation_w]
  const ActivationType activation_;
  const float relux_max_limit_;
  const float prelu_alpha_;
};

template <DeviceType D, typename T>
struct DepthwiseConv2dFunctor : public DepthwiseConv2dFunctorBase {
  DepthwiseConv2dFunctor(const int *strides,
                         const Padding padding,
                         const int *dilations,
                         const ActivationType activation,
                         const float relux_max_limit,
                         const float prelu_alpha)
      : DepthwiseConv2dFunctorBase(strides,
                                   padding,
                                   dilations,
                                   activation,
                                   relux_max_limit,
                                   prelu_alpha) {}

  void operator()(const Tensor *input,   // NHWC
                  const Tensor *filter,  // HWIM
                  const Tensor *bias,    // O
                  Tensor *output,
                  StatsFuture *future) {
    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    // Create a fake conv_2d filter to calculate the paddings and output size
    std::vector<index_t> fake_filter_shape(4);
    fake_filter_shape[0] = filter->shape()[0];
    fake_filter_shape[1] = filter->shape()[1];
    fake_filter_shape[3] = filter->shape()[2] * filter->shape()[3];
    fake_filter_shape[2] = 1;

    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);
    kernels::CalcNHWCPaddingAndOutputSize(
        input->shape().data(), fake_filter_shape.data(), dilations_, strides_,
        padding_, output_shape.data(), paddings.data());
    auto input_shape = fake_filter_shape;
    output->Resize(output_shape);

    index_t batch = output->dim(0);
    index_t height = output->dim(1);
    index_t width = output->dim(2);
    index_t channels = output->dim(3);

    index_t input_batch = input->dim(0);
    index_t input_height = input->dim(1);
    index_t input_width = input->dim(2);
    index_t input_channels = input->dim(3);

    index_t kernel_h = filter->dim(0);
    index_t kernel_w = filter->dim(1);
    index_t multiplier = filter->dim(3);
    MACE_CHECK(filter->dim(2) == input_channels, filter->dim(2), "!=", input_channels);
    MACE_CHECK(channels == input_channels * multiplier);

    int stride_h = strides_[0];
    int stride_w = strides_[1];

    int dilation_h = dilations_[0];
    int dilation_w = dilations_[1];

    MACE_CHECK(batch == input_batch, "Input/Output batch size mismatch");

    // The left-upper most offset of the padded input
    int padded_h_start = 0 - paddings[0] / 2;
    int padded_w_start = 0 - paddings[1] / 2;
    index_t padded_h_stop = input_height + paddings[0] - paddings[0] / 2;
    index_t padded_w_stop = input_width + paddings[1] - paddings[1] / 2;

    const index_t kernel_size = kernel_h * kernel_w;

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard filter_mapper(filter);
    Tensor::MappingGuard bias_mapper(bias);
    Tensor::MappingGuard output_mapper(output);
    const T *input_ptr = input->data<T>();
    const T *filter_ptr = filter->data<T>();
    const T *bias_ptr = bias == nullptr ? nullptr : bias->data<T>();
    T *output_ptr = output->mutable_data<T>();

#pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; ++n) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          for (int c = 0; c < channels; ++c) {
            const index_t inc = c / multiplier;
            const index_t m = c % multiplier;
            T bias_channel = bias_ptr ? bias_ptr[c] : 0;
            index_t offset = n * height * width * channels +
                             h * width * channels + w * channels + c;
            output_ptr[offset] = bias_channel;
            T sum = 0;
            const T *filter_base = filter_ptr + inc * multiplier + m;
            for (int kh = 0; kh < kernel_h; ++kh) {
              for (int kw = 0; kw < kernel_w; ++kw) {
                int inh = padded_h_start + h * stride_h + dilation_h * kh;
                int inw = padded_w_start + w * stride_w + dilation_w * kw;
                if (inh < 0 || inh >= input_height || inw < 0 ||
                    inw >= input_width) {
                  MACE_CHECK(inh >= padded_h_start && inh < padded_h_stop &&
                                 inw >= padded_w_start && inw < padded_w_stop,
                             "Out of range read from input: ", inh, ", ", inw);
                } else {
                  index_t input_offset =
                      n * input_height * input_width * input_channels +
                      inh * input_width * input_channels +
                      inw * input_channels + inc;
                  sum += input_ptr[input_offset] * filter_base[0]; // HWIM
                }
                filter_base += input_channels * multiplier;
              }
            }
            output_ptr[offset] += sum;
          }
        }
      }
    }

    output_ptr = output->mutable_data<T>();
    DoActivation(output_ptr, output_ptr, output->NumElements(), activation_,
                 relux_max_limit_, prelu_alpha_);
  }
};

template <>
void DepthwiseConv2dFunctor<DeviceType::NEON, float>::operator()(
    const Tensor *input,
    const Tensor *filter,
    const Tensor *bias,
    Tensor *output,
    StatsFuture *future);

template <typename T>
struct DepthwiseConv2dFunctor<DeviceType::OPENCL, T>
    : DepthwiseConv2dFunctorBase {
  DepthwiseConv2dFunctor(const int *strides,
                         const Padding padding,
                         const int *dilations,
                         const ActivationType activation,
                         const float relux_max_limit,
                         const float prelu_alpha)
      : DepthwiseConv2dFunctorBase(strides,
                                   padding,
                                   dilations,
                                   activation,
                                   relux_max_limit,
                                   prelu_alpha){}

  void operator()(const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_DEPTHWISE_CONV2D_H_
