//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_POOLING_H
#define MACE_KERNELS_POOLING_H

#include <limits>
#include "mace/core/tensor.h"

namespace mace {

enum PoolingType {
  AVG = 1, // avg_pool
  MAX = 2, // max_pool
};

namespace kernels {

template<DeviceType D, typename T>
class PoolingFunctor {
public:
  PoolingFunctor(const PoolingType pooling_type,
                 const int* kernels,
                 const int* strides,
                 const int* paddings,
                 const int* dilations)
  : pooling_type_(pooling_type),
    kernels_(kernels),
    strides_(strides),
    paddings_(paddings),
    dilations_(dilations) {}

  void operator()(const T* input,
                  const index_t* input_shape,
                  T* output,
                  const index_t* output_shape) {
    index_t batch    = output_shape[0];
    index_t channels = output_shape[1];
    index_t height   = output_shape[2];
    index_t width    = output_shape[3];

    index_t input_channels = input_shape[1];
    index_t input_height   = input_shape[2];
    index_t input_width    = input_shape[3];

    int kernel_h = kernels_[0];
    int kernel_w  = kernels_[1];

    int stride_h = strides_[0];
    int stride_w = strides_[1];

    int dilation_h = dilations_[0];
    int dilation_w = dilations_[1];

    // The left-upper most offset of the padded input
    int padded_h_start = 0 - paddings_[0] / 2;
    int padded_w_start = 0 - paddings_[1] / 2;
    int padded_h_stop = input_height + paddings_[0] - paddings_[0] / 2;
    int padded_w_stop = input_width + paddings_[1] - paddings_[0] / 2;

#pragma omp parallel for collpse(2)
    for (int n = 0; n < batch; ++n) {
      for (int c = 0; c < channels; ++c) {
        index_t out_offset = n * channels * height * width +
                             c * height * width;
        index_t in_offset = n * input_channels * input_height * input_width +
                            c * input_height * input_width;
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            T sum_or_max = 0;
            switch (pooling_type_) {
              case AVG:
                break;
              case MAX:
                sum_or_max = -std::numeric_limits<T>::max();
                break;
              default:
                MACE_CHECK(false, "Unsupported pooling type: ", pooling_type_);
            }
            for (int kh = 0; kh < kernel_h; ++kh) {
              for (int kw = 0; kw < kernel_w; ++kw) {
                int inh = padded_h_start + h * stride_h + dilation_h * kh;
                int inw = padded_w_start + w * stride_w + dilation_w * kw;
                if (inh >= 0 && inh < input_height &&
                    inw >= 0 && inw < input_width) {
                  index_t input_offset = in_offset +
                                         inh * input_width + inw;
                  switch (pooling_type_) {
                    case AVG:
                      sum_or_max += input[input_offset];
                      break;
                    case MAX:
                      sum_or_max = std::max(sum_or_max, input[input_offset]);
                      break;
                    default:
                      MACE_CHECK(false, "Unsupported pooling type: ",
                                 pooling_type_);
                  }
                }
              }
            }
            switch (pooling_type_) {
              case AVG:
                output[out_offset] = sum_or_max / (kernel_h * kernel_w);
                break;
              case MAX:
                output[out_offset] = sum_or_max;
                break;
              default:
                MACE_CHECK(false, "Unsupported pooling type: ", pooling_type_);
            }
            out_offset += 1;
          }
        }
      }
    }
  }

private:
  const PoolingType pooling_type_;
  const int* kernels_;
  const int* strides_;
  const int* paddings_;
  const int* dilations_;
};


} //  namespace kernels
} //  namespace mace

#endif //MACE_KERNELS_POOLING_H
