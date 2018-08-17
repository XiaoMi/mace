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

#ifndef MACE_KERNELS_DECONV_2D_H_
#define MACE_KERNELS_DECONV_2D_H_

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <algorithm>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/utils/utils.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

namespace deconv {

template<typename T>
void Deconv2dNCHW(const T *input,
                  const T *filter,
                  const T *bias,
                  const index_t *in_shape,
                  const index_t *out_shape,
                  const index_t *kernel_hw,
                  const int *strides,
                  const int *padding,
                  float *output) {
#pragma omp parallel for collapse(4)
  for (index_t b = 0; b < out_shape[0]; ++b) {
    for (index_t oc = 0; oc < out_shape[1]; ++oc) {
      for (index_t oh = 0; oh < out_shape[2]; ++oh) {
        for (index_t ow = 0; ow < out_shape[3]; ++ow) {
          index_t filter_start_y, filter_start_x;
          index_t start_x = std::max<int>(0, ow + strides[1] -1 - padding[1]);
          index_t start_y = std::max<int>(0, oh + strides[0] -1 - padding[0]);
          start_x /= strides[1];
          start_y /= strides[0];
          filter_start_x = padding[1] + strides[1] * start_x - ow;
          filter_start_y = padding[0] + strides[0] * start_y - oh;
          filter_start_x = kernel_hw[1] - 1 - filter_start_x;
          filter_start_y = kernel_hw[0] - 1 - filter_start_y;
          T out_value = 0;
          index_t out_pos =
              ((b * out_shape[1] + oc) * out_shape[2] + oh) * out_shape[3] + ow;
          for (index_t ic = 0; ic < in_shape[1]; ++ic) {
            for (index_t f_y = filter_start_y, ih = start_y;
                 f_y >= 0 && ih < in_shape[2]; f_y -= strides[0], ++ih) {
              for (index_t f_x = filter_start_x, iw = start_x;
                  f_x >= 0 && iw < in_shape[3]; f_x -= strides[1], ++iw) {
                  index_t weight_pos =
                      ((oc * in_shape[1] + ic) * kernel_hw[0] + f_y)
                          * kernel_hw[1] + f_x;
                  index_t in_pos =
                      ((b * in_shape[1] + ic) * in_shape[2] + ih)
                          * in_shape[3] + iw;
                  out_value += input[in_pos] * filter[weight_pos];
              }
            }
          }
          if (bias != nullptr)
            out_value += bias[oc];
          output[out_pos] = out_value;
        }
      }
    }
  }
}
}  // namespace deconv

struct Deconv2dFunctorBase {
  Deconv2dFunctorBase(const int *strides,
                      const Padding &padding_type,
                      const std::vector<int> &paddings,
                      const std::vector<index_t> &output_shape,
                      const ActivationType activation,
                      const float relux_max_limit,
                      const bool from_caffe)
      : strides_(strides),
        padding_type_(padding_type),
        paddings_(paddings),
        output_shape_(output_shape),
        activation_(activation),
        relux_max_limit_(relux_max_limit),
        from_caffe_(from_caffe) {}

  static void CalcDeconvOutputSize(
      const index_t *input_shape,   // NHWC
      const index_t *filter_shape,  // OIHW
      const int *strides,
      index_t *output_shape,
      const int *padding_size,
      const bool isNCHW = false) {
    MACE_CHECK_NOTNULL(output_shape);
    MACE_CHECK_NOTNULL(padding_size);
    MACE_CHECK_NOTNULL(input_shape);
    MACE_CHECK_NOTNULL(filter_shape);
    MACE_CHECK_NOTNULL(strides);

    const index_t output_channel = filter_shape[0];

    const index_t in_height = isNCHW ? input_shape[2] : input_shape[1];
    const index_t in_width = isNCHW ? input_shape[3] : input_shape[2];

    const index_t filter_h = filter_shape[2];
    const index_t filter_w = filter_shape[3];

    index_t out_height =
        (in_height - 1) * strides[0] + filter_h -padding_size[0];
    index_t out_width =
        (in_width - 1) * strides[1] + filter_w -padding_size[1];

    output_shape[0] = input_shape[0];
    if (isNCHW) {
      output_shape[1] = output_channel;
      output_shape[2] = out_height;
      output_shape[3] = out_width;
    } else {
      output_shape[1] = out_height;
      output_shape[2] = out_width;
      output_shape[3] = output_channel;
    }
  }

  static void CalcDeconvPaddingAndInputSize(
      const index_t *input_shape,   // NHWC
      const index_t *filter_shape,  // OIHW
      const int *strides,
      Padding padding,
      const index_t *output_shape,
      int *padding_size,
      const bool isNCHW = false) {
    MACE_CHECK_NOTNULL(output_shape);
    MACE_CHECK_NOTNULL(padding_size);
    MACE_CHECK_NOTNULL(input_shape);
    MACE_CHECK_NOTNULL(filter_shape);
    MACE_CHECK_NOTNULL(strides);

    const index_t in_height = isNCHW ? input_shape[2] : input_shape[1];
    const index_t in_width = isNCHW ? input_shape[3] : input_shape[2];

    const index_t out_height = isNCHW ? output_shape[2] : output_shape[1];
    const index_t out_width = isNCHW ? output_shape[3] : output_shape[2];

    const index_t extended_input_height = (in_height - 1) * strides[0] + 1;
    const index_t extended_input_width = (in_width - 1) * strides[1] + 1;

    const index_t filter_h = filter_shape[2];
    const index_t filter_w = filter_shape[3];

    index_t expected_input_height = 0, expected_input_width = 0;

    switch (padding) {
      case VALID:
        expected_input_height =
            (out_height - filter_h + strides[0]) / strides[0];
        expected_input_width =
            (out_width - filter_w + strides[1]) / strides[1];
        break;
      case SAME:
        expected_input_height =
            (out_height + strides[0] - 1) / strides[0];
        expected_input_width =
            (out_width + strides[1] - 1) / strides[1];
        break;
      default:
        MACE_CHECK(false, "Unsupported padding type: ", padding);
    }

    MACE_CHECK(expected_input_height == in_height,
               expected_input_height, "!=", in_height);
    MACE_CHECK(expected_input_width == in_width,
               expected_input_width, "!=", in_width);

    const int p_h = static_cast<int>(out_height +
        filter_h - 1 - extended_input_height);
    const int p_w = static_cast<int>(out_width +
        filter_w - 1 - extended_input_width);

    padding_size[0] = std::max<int>(0, p_h);
    padding_size[1] = std::max<int>(0, p_w);
  }

  const int *strides_;  // [stride_h, stride_w]
  const Padding padding_type_;
  std::vector<int> paddings_;
  std::vector<index_t> output_shape_;
  const ActivationType activation_;
  const float relux_max_limit_;
  const bool from_caffe_;
};

template <DeviceType D, typename T>
struct Deconv2dFunctor : Deconv2dFunctorBase {
  Deconv2dFunctor(const int *strides,
                  const Padding &padding_type,
                  const std::vector<int> &paddings,
                  const std::vector<index_t> &output_shape,
                  const ActivationType activation,
                  const float relux_max_limit,
                  const bool from_caffe)
      : Deconv2dFunctorBase(strides,
                            padding_type,
                            paddings,
                            output_shape,
                            activation,
                            relux_max_limit,
                            from_caffe) {}

  MaceStatus operator()(const Tensor *input,   // NCHW
                  const Tensor *filter,  // OIHW
                  const Tensor *bias,
                  const Tensor *output_shape_tensor,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_UNUSED(future);
    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    if (!from_caffe_) {  // tensorflow
      std::vector<index_t> output_shape(4);
      if (output_shape_.size() == 4) {
        output_shape[0] = output_shape_[0];
        output_shape[1] = output_shape_[3];
        output_shape[2] = output_shape_[1];
        output_shape[3] = output_shape_[2];
      } else {
        MACE_CHECK_NOTNULL(output_shape_tensor);
        MACE_CHECK(output_shape_tensor->size() == 4);
        Tensor::MappingGuard output_shape_mapper(output_shape_tensor);
        auto output_shape_data =
            output_shape_tensor->data<int32_t>();
        output_shape =
            std::vector<index_t>(output_shape_data, output_shape_data + 4);
      }
      paddings_.clear();
      paddings_ = std::vector<int>(2, 0);
      CalcDeconvPaddingAndInputSize(
          input->shape().data(),
          filter->shape().data(),
          strides_, padding_type_,
          output_shape.data(),
          paddings_.data(), true);
      MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    } else {  // caffe
      output_shape_.clear();
      output_shape_ = std::vector<index_t>(4, 0);
      CalcDeconvOutputSize(input->shape().data(),
                           filter->shape().data(),
                           strides_,
                           output_shape_.data(),
                           paddings_.data(), true);
      MACE_RETURN_IF_ERROR(output->Resize(output_shape_));
    }
    index_t kernel_h = filter->dim(2);
    index_t kernel_w = filter->dim(3);
    const index_t *in_shape = input->shape().data();
    const index_t *out_shape = output->shape().data();
    const index_t kernel_hw[2] = {kernel_h, kernel_w};

    MACE_CHECK(filter->dim(0) == out_shape[1], filter->dim(0), " != ",
               out_shape[1]);
    MACE_CHECK(filter->dim(1) == in_shape[1], filter->dim(1), " != ",
               in_shape[1]);
    MACE_CHECK(in_shape[0] == out_shape[0], "Input/Output batch size mismatch");
    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard filter_mapper(filter);
    Tensor::MappingGuard bias_mapper(bias);
    Tensor::MappingGuard output_mapper(output);
    auto input_data = input->data<T>();
    auto filter_data = filter->data<T>();
    auto bias_data = bias == nullptr ? nullptr : bias->data<T>();
    auto output_data = output->mutable_data<T>();
    int padding[2];
    padding[0] = (paddings_[0] + 1) >> 1;
    padding[1] = (paddings_[1] + 1) >> 1;
    deconv::Deconv2dNCHW(input_data,
                         filter_data,
                         bias_data,
                         in_shape,
                         out_shape,
                         kernel_hw,
                         strides_,
                         padding,
                         output_data);

    DoActivation(output_data,
                 output_data,
                 output->size(),
                 activation_,
                 relux_max_limit_);

    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
struct Deconv2dFunctor<DeviceType::GPU, T> : Deconv2dFunctorBase {
  Deconv2dFunctor(const int *strides,
                  const Padding &padding_type,
                  const std::vector<int> &paddings,
                  const std::vector<index_t> &output_shape,
                  const ActivationType activation,
                  const float relux_max_limit,
                  const bool from_caffe)
      : Deconv2dFunctorBase(strides,
                            padding_type,
                            paddings,
                            output_shape,
                            activation,
                            relux_max_limit,
                            from_caffe) {}

  MaceStatus operator()(const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  const Tensor *output_shape_tensor,
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

#endif  // MACE_KERNELS_DECONV_2D_H_
