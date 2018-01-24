//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_CONV_2D_H_
#define MACE_KERNELS_CONV_2D_H_

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/conv_pool_2d_util.h"

namespace mace {
namespace kernels {

struct Conv2dFunctorBase {
  Conv2dFunctorBase(const int *strides,
                    const Padding &paddings,
                    const int *dilations,
                    const ActivationType activation,
                    const float relux_max_limit,
                    const float prelu_alpha)
      : strides_(strides),
        paddings_(paddings),
        dilations_(dilations),
        activation_(activation),
        relux_max_limit_(relux_max_limit),
        prelu_alpha_(prelu_alpha) {}

  const int *strides_;    // [stride_h, stride_w]
  const Padding paddings_;
  const int *dilations_;  // [dilation_h, dilation_w]
  const ActivationType activation_;
  const float relux_max_limit_;
  const float prelu_alpha_;
};

template <DeviceType D, typename T>
struct Conv2dFunctor : Conv2dFunctorBase {
  Conv2dFunctor(const int *strides,
                const Padding &paddings,
                const int *dilations,
                const ActivationType activation,
                const float relux_max_limit,
                const float prelu_alpha)
      : Conv2dFunctorBase(strides,
                          paddings,
                          dilations,
                          activation,
                          relux_max_limit,
                          prelu_alpha) {}

  void operator()(const Tensor *input,  // NHWC
                  const Tensor *filter, // HWIO
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);
    kernels::CalcNHWCPaddingAndOutputSize(
        input->shape().data(), filter->shape().data(), dilations_, strides_,
        paddings_, output_shape.data(), paddings.data());
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

    index_t kernel_size = input_channels * kernel_h * kernel_w;

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard filter_mapper(filter);
    Tensor::MappingGuard bias_mapper(bias);
    Tensor::MappingGuard output_mapper(output);
    auto input_data = input->data<T>();
    auto filter_data = filter->data<T>();
    auto bias_data = bias == nullptr ? nullptr : bias->data<T>();
    auto output_data = output->mutable_data<T>();

    for (int n = 0; n < batch; ++n) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          for (int c = 0; c < channels; ++c) {
            T bias_channel = 0.0f;
            if (bias) bias_channel = bias_data[c];
            *output_data = bias_channel;
            T sum = 0.0f;
            const T *filter_ptr = filter_data + c;
            for (int kh = 0; kh < kernel_h; ++kh) {
              for (int kw = 0; kw < kernel_w; ++kw) {
                for (int inc = 0; inc < input_channels; ++inc) {
                  int inh = padded_h_start + h * stride_h + dilation_h * kh;
                  int inw = padded_w_start + w * stride_w + dilation_w * kw;
                  if (inh < 0 || inh >= input_height || inw < 0 ||
                      inw >= input_width) {
                    MACE_CHECK(inh >= padded_h_start && inh < padded_h_stop &&
                                   inw >= padded_w_start && inw < padded_w_stop,
                               "Out of range read from input: ", inh, ", ",
                               inw);
                    // else padding with 0:
                    // sum += 0;
                  } else {
                    index_t input_offset =
                        n * input_height * input_width * input_channels +
                        inh * input_width * input_channels +
                        inw * input_channels + inc;
                    sum += input_data[input_offset] * *filter_ptr;
                  }
                  filter_ptr += channels;
                }
              }
            }
            *output_data += sum;
            output_data++;
          }
        }
      }
    }
    output_data = output->mutable_data<T>();
    DoActivation(output_data, output_data, output->NumElements(), activation_,
                 relux_max_limit_, prelu_alpha_);
  }
};

template <>
void Conv2dFunctor<DeviceType::NEON, float>::operator()(const Tensor *input,
                                                        const Tensor *filter,
                                                        const Tensor *bias,
                                                        Tensor *output,
                                                        StatsFuture *future);

template <typename T>
struct Conv2dFunctor<DeviceType::OPENCL, T> : Conv2dFunctorBase {
  Conv2dFunctor(const int *strides,
                const Padding &paddings,
                const int *dilations,
                const ActivationType activation,
                const float relux_max_limit,
                const float prelu_alpha)
      : Conv2dFunctorBase(strides,
                          paddings,
                          dilations,
                          activation,
                          relux_max_limit,
                          prelu_alpha) {}

  void operator()(const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future);
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_CONV_2D_H_
