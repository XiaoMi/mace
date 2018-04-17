// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MACE_KERNELS_DEPTHWISE_CONV2D_H_
#define MACE_KERNELS_DEPTHWISE_CONV2D_H_

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <algorithm>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/kernels/activation.h"
#include "mace/public/mace.h"

namespace mace {
namespace kernels {

template <typename T>
void DepthwiseConv2dKernel(const T *input_ptr,
                           const T *filter_ptr,
                           const T *bias_ptr,
                           T *output_ptr,
                           int batch,
                           int height,
                           int width,
                           int channels,
                           int input_height,
                           int input_width,
                           int input_channels,
                           int multiplier,
                           int padded_h_start,
                           int padded_h_stop,
                           int padded_w_start,
                           int padded_w_stop,
                           int kernel_h,
                           int kernel_w,
                           int stride_h,
                           int stride_w,
                           int dilation_h,
                           int dilation_w,
                           int h_start,
                           int h_stop,
                           int w_start,
                           int w_stop) {
#pragma omp parallel for collapse(4)
  for (int n = 0; n < batch; ++n) {
    for (int h = h_start; h < h_stop; ++h) {
      for (int w = w_start; w < w_stop; ++w) {
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
                           "Out of range read from input: ", padded_h_start,
                           " <= ", inh, " < ", padded_h_stop, ", ",
                           padded_w_start, " <= ", inw, " < ", padded_w_stop);
              } else {
                index_t input_offset =
                    n * input_height * input_width * input_channels +
                    inh * input_width * input_channels + inw * input_channels +
                    inc;
                sum += input_ptr[input_offset] * filter_base[0];  // HWIM
              }
              filter_base += input_channels * multiplier;
            }
          }
          output_ptr[offset] += sum;
        }
      }
    }
  }
}
template <typename T>
void DepthwiseConv2dNoOOBCheckKernel(const T *input_ptr,
                                     const T *filter_ptr,
                                     const T *bias_ptr,
                                     T *output_ptr,
                                     int batch,
                                     int height,
                                     int width,
                                     int channels,
                                     int input_height,
                                     int input_width,
                                     int input_channels,
                                     int multiplier,
                                     int padded_h_start,
                                     int padded_h_stop,
                                     int padded_w_start,
                                     int padded_w_stop,
                                     int kernel_h,
                                     int kernel_w,
                                     int stride_h,
                                     int stride_w,
                                     int dilation_h,
                                     int dilation_w,
                                     int h_start,
                                     int h_stop,
                                     int w_start,
                                     int w_stop) {
  if (multiplier == 1) {
    constexpr int c_tile_size = 4;

#pragma omp parallel for collapse(3)
    for (int n = 0; n < batch; ++n) {
      for (int h = h_start; h < h_stop; ++h) {
        for (int w = w_start; w < w_stop; ++w) {
          int c;
          for (c = 0; c + c_tile_size <= channels; c += c_tile_size) {
#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
            static_assert(c_tile_size == 4, "channels tile size must be 4");
            float32x4_t sum = vdupq_n_f32(0);
            if (bias_ptr != nullptr) {
              sum = vld1q_f32(bias_ptr + c);
            }
#else
            T sum[c_tile_size] = {0};
            if (bias_ptr != nullptr) {
              for (int ci = 0; ci < c_tile_size; ++ci) {
                sum[ci] = bias_ptr[c + ci];
              }
            }
#endif
            const T *filter_base = filter_ptr + c;
            for (int kh = 0; kh < kernel_h; ++kh) {
              for (int kw = 0; kw < kernel_w; ++kw) {
                int inh = padded_h_start + h * stride_h + dilation_h * kh;
                int inw = padded_w_start + w * stride_w + dilation_w * kw;
                MACE_ASSERT(inh >= 0 && inh < input_height && inw >= 0 &&
                            inw < input_width);
                index_t input_offset =
                    n * input_height * input_width * input_channels +
                    inh * input_width * input_channels + inw * input_channels +
                    c;
#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
                float32x4_t in = vld1q_f32(input_ptr + input_offset);
                float32x4_t weights = vld1q_f32(filter_base);
                sum = vfmaq_f32(sum, in, weights);
#else
                for (int ci = 0; ci < c_tile_size; ++ci) {
                  sum[ci] +=
                      input_ptr[input_offset + ci] * filter_base[ci];  // HWIM
                }
#endif
                filter_base += input_channels;
              }
            }

            index_t offset = n * height * width * channels +
                             h * width * channels + w * channels + c;
#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
            vst1q_f32(output_ptr + offset, sum);
#else
            for (int ci = 0; ci < c_tile_size; ++ci) {
              output_ptr[offset + ci] = sum[ci];
            }
#endif
          }
          for (; c < channels; ++c) {
            T bias_channel = bias_ptr ? bias_ptr[c] : 0;
            index_t offset = n * height * width * channels +
                             h * width * channels + w * channels + c;
            output_ptr[offset] = bias_channel;
            T sum = 0;
            const T *filter_base = filter_ptr + c;
            for (int kh = 0; kh < kernel_h; ++kh) {
              for (int kw = 0; kw < kernel_w; ++kw) {
                int inh = padded_h_start + h * stride_h + dilation_h * kh;
                int inw = padded_w_start + w * stride_w + dilation_w * kw;
                MACE_ASSERT(inh >= 0 && inh < input_height && inw >= 0 &&
                            inw < input_width);
                index_t input_offset =
                    n * input_height * input_width * input_channels +
                    inh * input_width * input_channels + inw * input_channels +
                    c;
                sum += input_ptr[input_offset] * filter_base[0];  // HWIM
                filter_base += input_channels * multiplier;
              }
            }
            output_ptr[offset] += sum;
          }
        }
      }
    }
  } else {
#pragma omp parallel for collapse(4)
    for (int n = 0; n < batch; ++n) {
      for (int h = h_start; h < h_stop; ++h) {
        for (int w = w_start; w < w_stop; ++w) {
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
                MACE_ASSERT(inh >= 0 && inh < input_height && inw >= 0 &&
                            inw < input_width);
                index_t input_offset =
                    n * input_height * input_width * input_channels +
                    inh * input_width * input_channels + inw * input_channels +
                    inc;
                sum += input_ptr[input_offset] * filter_base[0];  // HWIM
                filter_base += input_channels * multiplier;
              }
            }
            output_ptr[offset] += sum;
          }
        }
      }
    }
  }
}

struct DepthwiseConv2dFunctorBase {
  DepthwiseConv2dFunctorBase(const int *strides,
                             const Padding padding_type,
                             const std::vector<int> &paddings,
                             const int *dilations,
                             const ActivationType activation,
                             const float relux_max_limit)
      : strides_(strides),
        padding_type_(padding_type),
        paddings_(paddings),
        dilations_(dilations),
        activation_(activation),
        relux_max_limit_(relux_max_limit) {}

  const int *strides_;  // [stride_h, stride_w]
  const Padding padding_type_;
  std::vector<int> paddings_;
  const int *dilations_;  // [dilation_h, dilation_w]
  const ActivationType activation_;
  const float relux_max_limit_;
};

template <DeviceType D, typename T>
struct DepthwiseConv2dFunctor : public DepthwiseConv2dFunctorBase {
  DepthwiseConv2dFunctor(const int *strides,
                         const Padding padding_type,
                         const std::vector<int> &paddings,
                         const int *dilations,
                         const ActivationType activation,
                         const float relux_max_limit)
      : DepthwiseConv2dFunctorBase(strides,
                                   padding_type,
                                   paddings,
                                   dilations,
                                   activation,
                                   relux_max_limit) {}

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
    fake_filter_shape[2] = filter->shape()[2] * filter->shape()[3];
    fake_filter_shape[3] = 1;

    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);
    if (paddings_.empty()) {
      kernels::CalcNHWCPaddingAndOutputSize(
          input->shape().data(), fake_filter_shape.data(), dilations_, strides_,
          padding_type_, output_shape.data(), paddings.data());
    } else {
      paddings = paddings_;
      CalcOutputSize(input->shape().data(), fake_filter_shape.data(),
                     paddings_.data(), dilations_, strides_, RoundType::FLOOR,
                     output_shape.data());
    }
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
    MACE_CHECK(filter->dim(2) == input_channels, filter->dim(2), "!=",
               input_channels);
    MACE_CHECK(channels == input_channels * multiplier);

    int stride_h = strides_[0];
    int stride_w = strides_[1];

    int dilation_h = dilations_[0];
    int dilation_w = dilations_[1];

    MACE_CHECK(batch == input_batch, "Input/Output batch size mismatch");

    // The left-upper most offset of the padded input
    int paddings_top = paddings[0] / 2;
    int paddings_bottom = paddings[0] - paddings_top;
    int paddings_left = paddings[1] / 2;
    int paddings_right = paddings[1] - paddings_left;

    int padded_h_start = 0 - paddings_top;
    int padded_w_start = 0 - paddings_left;
    index_t padded_h_stop = input_height + paddings_bottom;
    index_t padded_w_stop = input_width + paddings_right;

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard filter_mapper(filter);
    Tensor::MappingGuard bias_mapper(bias);
    Tensor::MappingGuard output_mapper(output);
    const T *input_ptr = input->data<T>();
    const T *filter_ptr = filter->data<T>();
    const T *bias_ptr = bias == nullptr ? nullptr : bias->data<T>();
    T *output_ptr = output->mutable_data<T>();

    int valid_h_start =
        paddings_top == 0 ? 0 : (paddings_top - 1) / stride_h + 1;
    int valid_h_stop = paddings_bottom == 0
                           ? height
                           : height - ((paddings_bottom - 1) / stride_h + 1);
    int valid_w_start =
        paddings_left == 0 ? 0 : (paddings_left - 1) / stride_w + 1;
    int valid_w_stop = paddings_right == 0
                           ? width
                           : width - ((paddings_right - 1) / stride_w + 1);

    // Calculate border elements with out-of-boundary checking
    if (valid_h_start > 0) {
      DepthwiseConv2dKernel<T>(
          input_ptr, filter_ptr, bias_ptr, output_ptr, batch, height, width,
          channels, input_height, input_width, input_channels, multiplier,
          padded_h_start, padded_h_stop, padded_w_start, padded_w_stop,
          kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w, 0,
          valid_h_start, 0, width);
    }
    if (valid_h_stop < height) {
      DepthwiseConv2dKernel<T>(
          input_ptr, filter_ptr, bias_ptr, output_ptr, batch, height, width,
          channels, input_height, input_width, input_channels, multiplier,
          padded_h_start, padded_h_stop, padded_w_start, padded_w_stop,
          kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w,
          std::max(valid_h_start, valid_h_stop), height, 0, width);
    }
    if (valid_w_start > 0) {
      DepthwiseConv2dKernel<T>(
          input_ptr, filter_ptr, bias_ptr, output_ptr, batch, height, width,
          channels, input_height, input_width, input_channels, multiplier,
          padded_h_start, padded_h_stop, padded_w_start, padded_w_stop,
          kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w,
          valid_h_start, valid_h_stop, 0, valid_w_start);
    }
    if (valid_w_stop < width) {
      DepthwiseConv2dKernel<T>(
          input_ptr, filter_ptr, bias_ptr, output_ptr, batch, height, width,
          channels, input_height, input_width, input_channels, multiplier,
          padded_h_start, padded_h_stop, padded_w_start, padded_w_stop,
          kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w,
          valid_h_start, valid_h_stop, std::max(valid_w_start, valid_w_stop),
          width);
    }

    // Calculate border elements without out-of-boundary checking
    DepthwiseConv2dNoOOBCheckKernel<T>(
        input_ptr, filter_ptr, bias_ptr, output_ptr, batch, height, width,
        channels, input_height, input_width, input_channels, multiplier,
        padded_h_start, padded_h_stop, padded_w_start, padded_w_stop, kernel_h,
        kernel_w, stride_h, stride_w, dilation_h, dilation_w, valid_h_start,
        valid_h_stop, valid_w_start, valid_w_stop);

    output_ptr = output->mutable_data<T>();
    DoActivation(output_ptr, output_ptr, output->size(), activation_,
                 relux_max_limit_);
  }
};

template <>
struct DepthwiseConv2dFunctor<DeviceType::NEON, float>
  : DepthwiseConv2dFunctorBase {
  DepthwiseConv2dFunctor(const int *strides,
                         const Padding padding_type,
                         const std::vector<int> &paddings,
                         const int *dilations,
                         const ActivationType activation,
                         const float relux_max_limit)
    : DepthwiseConv2dFunctorBase(strides,
                                 padding_type,
                                 paddings,
                                 dilations,
                                 activation,
                                 relux_max_limit) {}

  void operator()(const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future);
};

template <typename T>
struct DepthwiseConv2dFunctor<DeviceType::OPENCL, T>
    : DepthwiseConv2dFunctorBase {
  DepthwiseConv2dFunctor(const int *strides,
                         const Padding padding_type,
                         const std::vector<int> &paddings,
                         const int *dilations,
                         const ActivationType activation,
                         const float relux_max_limit)
      : DepthwiseConv2dFunctorBase(strides,
                                   padding_type,
                                   paddings,
                                   dilations,
                                   activation,
                                   relux_max_limit) {}

  void operator()(const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_DEPTHWISE_CONV2D_H_
