//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_DEPTHWISE_CONV_H_
#define MACE_KERNELS_DEPTHWISE_CONV_H_

#include "mace/core/common.h"
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/proto/mace.pb.h"

namespace mace {
namespace kernels {

template <DeviceType D, typename T>
struct DepthwiseConv2dFunctor {
  DepthwiseConv2dFunctor() {}
  DepthwiseConv2dFunctor(const int *strides,
                         const std::vector<int> &paddings,
                         const int *dilations)
      : strides_(strides), paddings_(paddings), dilations_(dilations) {}

  void operator()(const Tensor *input,  // NCHW
                  const Tensor *filter,  // c_out, c_in, kernel_h, kernel_w
                  const Tensor *bias,  // c_out
                  Tensor *output) {
    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(bias);
    MACE_CHECK_NOTNULL(output);

    index_t batch = output->dim(0);
    index_t channels = output->dim(1);
    index_t height = output->dim(2);
    index_t width = output->dim(3);

    index_t input_batch = input->dim(0);
    index_t input_channels = input->dim(1);
    index_t input_height = input->dim(2);
    index_t input_width = input->dim(3);

    index_t kernel_h = filter->dim(2);
    index_t kernel_w = filter->dim(3);

    int stride_h = strides_[0];
    int stride_w = strides_[1];

    int dilation_h = dilations_[0];
    int dilation_w = dilations_[1];

    MACE_CHECK(batch == input_batch, "Input/Output batch size mismatch");

    // The left-upper most offset of the padded input
    int padded_h_start = 0 - paddings_[0] / 2;
    int padded_w_start = 0 - paddings_[1] / 2;
    index_t padded_h_stop = input_height + paddings_[0] - paddings_[0] / 2;
    index_t padded_w_stop = input_width + paddings_[1] - paddings_[1] / 2;

    index_t kernel_size = kernel_h * kernel_w;
    index_t multiplier = filter->dim(0);

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard filter_mapper(filter);
    Tensor::MappingGuard bias_mapper(bias);
    Tensor::MappingGuard output_mapper(output);
    const T *input_ptr = input->data<T>();
    const T *filter_ptr = filter->data<T>();
    const T *bias_ptr   = bias->data<T>();
    T *output_ptr = output->mutable_data<T>();

#pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; ++n) {
      for (int c = 0; c < channels; ++c) {
        T bias_channel = bias_ptr ? bias_ptr[c] : 0;
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            index_t offset = n * channels * height * width +
                             c * height * width + h * width + w;
            output_ptr[offset] = bias_channel;
            T sum = 0;
            const T *filter_base = filter_ptr + c * kernel_size;
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
                      n * input_channels * input_height * input_width +
                      (c / multiplier) * input_height * input_width +
                      inh * input_width + inw;
                  sum += input_ptr[input_offset] * *filter_base;
                }
                ++filter_base;
              }
            }
            output_ptr[offset] += sum;
          }
        }
      }
    }
  }

  const int *strides_;         // [stride_h, stride_w]
  std::vector<int> paddings_;  // [padding_h, padding_w]
  const int *dilations_;       // [dilation_h, dilation_w]
};

template <>
void DepthwiseConv2dFunctor<DeviceType::NEON, float>::operator()(
    const Tensor *input,
    const Tensor *filter,
    const Tensor *bias,
    Tensor *output);

template <>
void DepthwiseConv2dFunctor<DeviceType::OPENCL, float>::operator()(
    const Tensor *input,
    const Tensor *filter,
    const Tensor *bias,
    Tensor *output);

}  //  namespace kernels
}  //  namespace mace

#endif  //  MACE_KERNELS_DEPTHWISE_CONV_H_
