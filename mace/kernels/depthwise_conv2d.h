//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_DEPTHWISE_CONV_H_
#define MACE_KERNELS_DEPTHWISE_CONV_H_

#include "mace/proto/mace.pb.h"
#include "mace/core/common.h"
#include "mace/kernels/conv_pool_2d_util.h"

namespace mace {
namespace kernels {

template<DeviceType D, typename T>
class DepthwiseConv2dFunctor {
 public:
  DepthwiseConv2dFunctor(const index_t* input_shape,
                         const index_t* filter_shape,
                         const int* strides,
                         const Padding padding,
                         const int* dilations) :
      strides_(strides),
      paddings_(2, 0),
      dilations_(dilations) {
    CalPaddingSize(input_shape, filter_shape, dilations_, strides_, padding, paddings_.data());
  }
  DepthwiseConv2dFunctor(const int* strides,
                         const std::vector<int>& paddings,
                         const int* dilations) :
      strides_(strides),
      paddings_(paddings),
      dilations_(dilations) {}

  void operator()(const T* input, // NCHW
                  const index_t* input_shape,
                  const T* filter, // c_out, c_in, kernel_h, kernel_w
                  const index_t* filter_shape,
                  const T* bias, // c_out
                  T* output, // NCHW
                  const index_t* output_shape) {

    MACE_CHECK_NOTNULL(output);

    index_t batch = output_shape[0];
    index_t channels = output_shape[1];
    index_t height = output_shape[2];
    index_t width = output_shape[3];

    index_t input_batch = input_shape[0];
    index_t input_channels = input_shape[1];
    index_t input_height = input_shape[2];
    index_t input_width = input_shape[3];

    index_t kernel_h = filter_shape[2];
    index_t kernel_w = filter_shape[3];

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

    index_t kernel_size = filter_shape[1] * kernel_h * kernel_w;
    index_t multiplier = channels / input_channels;

#pragma omp parallel for collapse(2)
    for (int n = 0; n < batch; ++n) {
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            index_t offset = n * channels * height * width +
                c * height * width + h * width + w;
            T sum = 0;
            const T* filter_ptr = filter + c * kernel_size;
            for (int kh = 0; kh < kernel_h; ++kh) {
              for (int kw = 0; kw < kernel_w; ++kw) {
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
                      n * input_channels * input_height * input_width +
                          (c / multiplier) * input_height * input_width + inh * input_width +
                          inw;
                  sum += input[input_offset] * *filter_ptr;
                }
                ++filter_ptr;
              }
            }
            output[offset] = sum + bias[c];
          }
        }
      }
    }
  }
 private:
  const int* strides_; // [stride_h, stride_w]
  std::vector<int> paddings_;   // [padding_h, padding_w]
  const int* dilations_; // [dilation_h, dilation_w]
};

template <>
void DepthwiseConv2dFunctor<DeviceType::NEON, float>::operator()(const float* input,
                                                        const index_t* input_shape,
                                                        const float* filter,
                                                        const index_t* filter_shape,
                                                        const float* bias,
                                                        float* output,
                                                        const index_t* output_shape);
} //  namespace kernels
} //  namespace mace

#endif //  MACE_KERNELS_DEPTHWISE_CONV_H_
