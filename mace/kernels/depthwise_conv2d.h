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
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/arm/depthwise_conv2d_neon.h"
#include "mace/public/mace.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

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

template<DeviceType D, typename T>
struct DepthwiseConv2dFunctor;

template<>
struct DepthwiseConv2dFunctor<DeviceType::CPU, float>
  : public DepthwiseConv2dFunctorBase {
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

  void DepthwiseConv2dGeneral(const float *input,
                              const float *filter,
                              const index_t *in_shape,
                              const index_t *out_shape,
                              const index_t *filter_shape,
                              const int *stride_hw,
                              const int *dilation_hw,
                              const int *pad_hw,
                              float *output) {
    const index_t multiplier = filter_shape[0] / filter_shape[1];
#pragma omp parallel for collapse(2)
    for (index_t b = 0; b < in_shape[0]; ++b) {
      for (index_t m = 0; m < filter_shape[0]; ++m) {
        for (index_t h = 0; h < out_shape[2]; ++h) {
          for (index_t w = 0; w < out_shape[3]; ++w) {
            const index_t out_channels = filter_shape[0];
            const index_t in_channels = filter_shape[1];
            const index_t filter_height = filter_shape[2];
            const index_t filter_width = filter_shape[3];
            const index_t in_height = in_shape[2];
            const index_t in_width = in_shape[3];
            const index_t out_height = out_shape[2];
            const index_t out_width = out_shape[3];
            index_t out_offset =
              ((b * out_channels + m) * out_height + h) * out_width + w;
            index_t c = m / multiplier;
            index_t o = m % multiplier;
            float sum = 0;
            for (index_t kh = 0; kh < filter_height; ++kh) {
              for (index_t kw = 0; kw < filter_width; ++kw) {
                index_t ih = h * stride_hw[0] + kh * dilation_hw[0] - pad_hw[0];
                index_t iw = w * stride_hw[1] + kw * dilation_hw[1] - pad_hw[1];
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                  index_t in_offset =
                    ((b * in_channels + c) * in_height + ih) * in_width + iw;
                  index_t filter_offset =
                    (((o * in_channels) + c) * filter_height + kh)
                      * filter_width
                      + kw;

                  sum += input[in_offset] * filter[filter_offset];
                }
              }
            }
            output[out_offset] = sum;
          }
        }
      }
    }
  }

  MaceStatus operator()(const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_UNUSED(future);
    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);
    std::vector<index_t> filter_shape
      {filter->dim(0) * filter->dim(1), filter->dim(1), filter->dim(2),
       filter->dim(3)};

    if (paddings_.empty()) {
      CalcNCHWPaddingAndOutputSize(input->shape().data(),
                                   filter_shape.data(),
                                   dilations_,
                                   strides_,
                                   padding_type_,
                                   output_shape.data(),
                                   paddings.data());
    } else {
      paddings = paddings_;
      CalcNCHWOutputSize(input->shape().data(),
                         filter_shape.data(),
                         paddings_.data(),
                         dilations_,
                         strides_,
                         RoundType::FLOOR,
                         output_shape.data());
    }
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    output->Clear();

    index_t batch = output->dim(0);
    index_t channels = output->dim(1);
    index_t height = output->dim(2);
    index_t width = output->dim(3);

    index_t input_batch = input->dim(0);
    index_t input_channels = input->dim(1);
    index_t input_height = input->dim(2);
    index_t input_width = input->dim(3);

    index_t filter_h = filter_shape[2];
    index_t filter_w = filter_shape[3];
    MACE_CHECK(filter_shape[0] == channels, filter_shape[0], " != ", channels);
    MACE_CHECK(filter_shape[1] == input_channels, filter_shape[1], " != ",
               input_channels);

    index_t stride_h = strides_[0];
    index_t stride_w = strides_[1];

    index_t dilation_h = dilations_[0];
    index_t dilation_w = dilations_[1];

    MACE_CHECK(batch == input_batch, "Input/Output batch size mismatch");

    int pad_top = paddings[0] >> 1;
    int pad_bottom = paddings[0] - pad_top;
    int pad_left = paddings[1] >> 1;
    int pad_right = paddings[1] - pad_left;

    index_t valid_h_start = pad_top == 0 ? 0 : (pad_top - 1) / stride_h + 1;
    index_t valid_h_stop = pad_bottom == 0
                           ? height
                           : height - ((pad_bottom - 1) / stride_h + 1);
    index_t valid_w_start = pad_left == 0 ? 0 : (pad_left - 1) / stride_w + 1;
    index_t valid_w_stop = pad_right == 0
                           ? width
                           : width - ((pad_right - 1) / stride_w + 1);

    std::function<void(const float *input, float *output)> conv_func;

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard filter_guard(filter);
    Tensor::MappingGuard bias_guard(bias);
    Tensor::MappingGuard output_guard(output);
    auto input_data = input->data<float>();
    auto filter_data = filter->data<float>();
    auto bias_data = bias == nullptr ? nullptr : bias->data<float>();
    auto output_data = output->mutable_data<float>();

    const int pad_hw[2] = {pad_top, pad_left};
    const index_t input_shape[4] =
        {batch, input_channels, input_height, input_width};

    if (filter_h == 3 && filter_w == 3 && stride_h == 1 && stride_w == 1
      && dilation_h == 1 && dilation_w == 1) {
      conv_func = [=](const float *input, float *output) {
        DepthwiseConv2dNeonK3x3S1(input,
                                  filter_data,
                                  input_shape,
                                  output_shape.data(),
                                  pad_hw,
                                  valid_h_start,
                                  valid_h_stop,
                                  valid_w_start,
                                  valid_w_stop,
                                  output);
      };
    } else if (filter_h == 3 && filter_w == 3 && stride_h == 2 && stride_w == 2
      && dilation_h == 1 && dilation_w == 1) {
      conv_func = [=](const float *input, float *output) {
        DepthwiseConv2dNeonK3x3S2(input,
                                  filter_data,
                                  input_shape,
                                  output_shape.data(),
                                  pad_hw,
                                  valid_h_start,
                                  valid_h_stop,
                                  valid_w_start,
                                  valid_w_stop,
                                  output);
      };
    } else {
      conv_func = [=](const float *input, float *output) {
        DepthwiseConv2dGeneral(input,
                               filter_data,
                               input_shape,
                               output_shape.data(),
                               filter_shape.data(),
                               strides_,
                               dilations_,
                               pad_hw,
                               output);
      };
    }

    conv_func(input_data, output_data);

    if (bias_data != nullptr) {
#pragma omp parallel for collapse(2)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channels; ++c) {
          for (index_t i = 0; i < height * width; ++i) {
            output_data[(b * channels + c) * height * width + i] +=
              bias_data[c];
          }
        }
      }
    }

    DoActivation(output_data, output_data, output->size(), activation_,
                 relux_max_limit_);

    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template<typename T>
struct DepthwiseConv2dFunctor<DeviceType::GPU, T>
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

  MaceStatus operator()(const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_DEPTHWISE_CONV2D_H_
