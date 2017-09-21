//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_POOLING_H
#define MACE_KERNELS_POOLING_H

#include <limits>
#include "mace/core/tensor.h"

namespace mace {

enum PoolingType {
  AVG = 1,  // avg_pool
  MAX = 2,  // max_pool
};

namespace kernels {

template <DeviceType D, typename T>
struct PoolingFunctor {
  PoolingFunctor(const PoolingType pooling_type,
                 const int *kernels,
                 const int *strides,
                 const int *paddings,
                 const int *dilations)
      : pooling_type_(pooling_type),
        kernels_(kernels),
        strides_(strides),
        paddings_(paddings),
        dilations_(dilations) {}

  void operator()(const T *input,
                  const index_t *input_shape,
                  T *output,
                  const index_t *output_shape) {
    index_t batch = output_shape[0];
    index_t channels = output_shape[1];
    index_t height = output_shape[2];
    index_t width = output_shape[3];
    index_t out_image_size = height * width;

    index_t input_channels = input_shape[1];
    index_t input_height = input_shape[2];
    index_t input_width = input_shape[3];
    index_t in_image_size = input_height * input_width;

    int kernel_h = kernels_[0];
    int kernel_w = kernels_[1];

    int stride_h = strides_[0];
    int stride_w = strides_[1];

    int dilation_h = dilations_[0];
    int dilation_w = dilations_[1];

    // The left-upper most offset of the padded input
    int padded_h_start = 0 - paddings_[0] / 2;
    int padded_w_start = 0 - paddings_[1] / 2;

    if (pooling_type_ == MAX) {
#pragma omp parallel for collapse(2)
      for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
          index_t out_offset = (b * channels + c) * out_image_size;
          index_t in_offset = (b * input_channels + c) * in_image_size;
          for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
              T max = std::numeric_limits<T>::lowest();
              for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                  int inh = padded_h_start + h * stride_h + dilation_h * kh;
                  int inw = padded_w_start + w * stride_w + dilation_w * kw;
                  if (inh >= 0 && inh < input_height && inw >= 0 &&
                      inw < input_width) {
                    index_t input_offset = in_offset + inh * input_width + inw;
                    max = std::max(max, input[input_offset]);
                  }
                }
              }
              output[out_offset] = max;
              out_offset += 1;
            }
          }
        }
      }
    } else if (pooling_type_ == AVG) {
#pragma omp parallel for collapse(2)
      for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
          index_t out_offset = (b * channels + c) * out_image_size;
          index_t in_offset = (b * input_channels + c) * in_image_size;
          for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
              T sum = 0;
              for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                  int inh = padded_h_start + h * stride_h + dilation_h * kh;
                  int inw = padded_w_start + w * stride_w + dilation_w * kw;
                  if (inh >= 0 && inh < input_height && inw >= 0 &&
                      inw < input_width) {
                    index_t input_offset = in_offset + inh * input_width + inw;
                    sum += input[input_offset];
                  }
                }
              }
              output[out_offset] = sum / (kernel_h * kernel_w);
              out_offset += 1;
            }
          }
        }
      }
    }
  }

  const PoolingType pooling_type_;
  const int *kernels_;
  const int *strides_;
  const int *paddings_;
  const int *dilations_;
};

template <>
void PoolingFunctor<DeviceType::NEON, float>::operator()(
    const float *input,
    const index_t *input_shape,
    float *output,
    const index_t *output_shape);

}  //  namespace kernels
}  //  namespace mace

#endif  // MACE_KERNELS_POOLING_H
