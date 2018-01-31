//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_POOLING_H
#define MACE_KERNELS_POOLING_H

#include <limits>
#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/core/runtime/opencl/cl2_header.h"

namespace mace {

enum PoolingType {
  AVG = 1,  // avg_pool
  MAX = 2,  // max_pool
};

namespace kernels {

struct PoolingFunctorBase {
  PoolingFunctorBase(const PoolingType pooling_type,
                     const int *kernels,
                     const int *strides,
                     const Padding padding,
                     const int *dilations)
      : pooling_type_(pooling_type),
        kernels_(kernels),
        strides_(strides),
        padding_(padding),
        dilations_(dilations) {}

  const PoolingType pooling_type_;
  const int *kernels_;
  const int *strides_;
  const Padding padding_;
  const int *dilations_;
};

template<DeviceType D, typename T>
struct PoolingFunctor : PoolingFunctorBase {
  PoolingFunctor(const PoolingType pooling_type,
                 const int *kernels,
                 const int *strides,
                 const Padding padding,
                 const int *dilations)
      : PoolingFunctorBase(pooling_type, kernels,
                           strides, padding,
                           dilations) {}

  void operator()(const Tensor *input_tensor,
                  Tensor *output_tensor,
                  StatsFuture *future) {

    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);
    std::vector<index_t> filter_shape = {
        kernels_[0], kernels_[1],
        input_tensor->dim(3), input_tensor->dim(3)
    };

    kernels::CalcNHWCPaddingAndOutputSize(
        input_tensor->shape().data(), filter_shape.data(),
        dilations_, strides_, this->padding_,
        output_shape.data(), paddings.data());
    output_tensor->Resize(output_shape);

    Tensor::MappingGuard in_guard(input_tensor);
    Tensor::MappingGuard out_guard(output_tensor);
    const T *input = input_tensor->data<T>();
    T *output = output_tensor->mutable_data<T>();
    const index_t *input_shape = input_tensor->shape().data();
    index_t batch = output_shape[0];
    index_t height = output_shape[1];
    index_t width = output_shape[2];
    index_t channels = output_shape[3];

    index_t input_height = input_shape[1];
    index_t input_width = input_shape[2];
    index_t input_channels = input_shape[3];
    index_t in_image_size = input_height * input_width;

    int kernel_h = kernels_[0];
    int kernel_w = kernels_[1];

    int stride_h = strides_[0];
    int stride_w = strides_[1];

    int dilation_h = dilations_[0];
    int dilation_w = dilations_[1];

    // The left-upper most offset of the padded input
    int padded_h_start = 0 - paddings[0] / 2;
    int padded_w_start = 0 - paddings[1] / 2;

    if (pooling_type_ == MAX) {
#pragma omp parallel for collapse(2)
      for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
              index_t in_offset = b * in_image_size * input_channels + c;
              T res = std::numeric_limits<T>::lowest();
              for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                  int inh = padded_h_start + h * stride_h + dilation_h * kh;
                  int inw = padded_w_start + w * stride_w + dilation_w * kw;
                  if (inh >= 0 && inh < input_height && inw >= 0 &&
                      inw < input_width) {
                    index_t input_offset = in_offset + (inh * input_width + inw) * input_channels;
                    res = std::max(res, input[input_offset]);
                  }
                }
              }
              *output = res;
              output++;
            }
          }
        }
      }
    } else if (pooling_type_ == AVG) {
#pragma omp parallel for collapse(2)
      for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
              index_t in_offset = b * in_image_size * input_channels + c;
              T sum = 0;
              int block_size = 0;
              for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                  int inh = padded_h_start + h * stride_h + dilation_h * kh;
                  int inw = padded_w_start + w * stride_w + dilation_w * kw;
                  if (inh >= 0 && inh < input_height && inw >= 0 &&
                      inw < input_width) {
                    index_t input_offset = in_offset + (inh * input_width + inw) * input_channels;
                    sum += input[input_offset];
                    block_size += 1;
                  }
                }
              }
              *output = sum / block_size;
              output++;
            }
          }
        }
      }
    }
  }

};

template<>
void PoolingFunctor<DeviceType::NEON, float>::operator()(
    const Tensor *input_tensor,
    Tensor *output_tensor,
    StatsFuture *future);

template<typename T>
struct PoolingFunctor<DeviceType::OPENCL, T> : PoolingFunctorBase {
  PoolingFunctor(const PoolingType pooling_type,
                 const int *kernels,
                 const int *strides,
                 const Padding padding,
                 const int *dilations)
      : PoolingFunctorBase(pooling_type, kernels,
                           strides, padding,
                           dilations) {}
  void operator()(const Tensor *input_tensor,
                  Tensor *output_tensor,
                  StatsFuture *future);

  cl::Kernel kernel_;
};

}  //  namespace kernels
}  //  namespace mace

#endif  // MACE_KERNELS_POOLING_H
