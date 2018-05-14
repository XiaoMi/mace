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

#ifndef MACE_KERNELS_POOLING_H_
#define MACE_KERNELS_POOLING_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/conv_pool_2d_util.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

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
                     const Padding padding_type,
                     const std::vector<int> &paddings,
                     const int *dilations)
      : pooling_type_(pooling_type),
        kernels_(kernels),
        strides_(strides),
        padding_type_(padding_type),
        paddings_(paddings),
        dilations_(dilations) {}

  const PoolingType pooling_type_;
  const int *kernels_;
  const int *strides_;
  const Padding padding_type_;
  std::vector<int> paddings_;
  const int *dilations_;
};

template <DeviceType D, typename T>
struct PoolingFunctor;

template <>
struct PoolingFunctor<DeviceType::CPU, float>: PoolingFunctorBase {
  PoolingFunctor(const PoolingType pooling_type,
                 const int *kernels,
                 const int *strides,
                 const Padding padding_type,
                 const std::vector<int> &paddings,
                 const int *dilations)
      : PoolingFunctorBase(
            pooling_type, kernels, strides, padding_type, paddings, dilations) {
  }

  void MaxPooling(const float *input,
                  const index_t batch,
                  const index_t in_height,
                  const index_t in_width,
                  const index_t channels,
                  const index_t out_height,
                  const index_t out_width,
                  const int filter_height,
                  const int filter_width,
                  const int stride_h,
                  const int stride_w,
                  const int dilation_h,
                  const int dilation_w,
                  const int pad_top,
                  const int pad_left,
                  float *output) {
    const index_t in_image_size = in_height * in_width;
    const index_t out_image_size = out_height * out_width;
    const index_t in_batch_size = channels * in_image_size;
    const index_t out_batch_size = channels * out_image_size;

#pragma omp parallel for collapse(2)
    for (index_t b = 0; b < batch; ++b) {
      for (index_t c = 0; c < channels; ++c) {
        const index_t out_base = b * out_batch_size + c * out_image_size;
        const index_t in_base = b * in_batch_size + c * in_image_size;
        for (index_t h = 0; h < out_height; ++h) {
          for (index_t w = 0; w < out_width; ++w) {
            const index_t out_offset = out_base + h * out_width + w;
            float res = std::numeric_limits<float>::lowest();
            for (int fh = 0; fh < filter_height; ++fh) {
              for (int fw = 0; fw < filter_width; ++fw) {
                int inh = h * stride_h + dilation_h * fh - pad_top;
                int inw = w * stride_w + dilation_w * fw - pad_left;
                if (inh >= 0 && inh < in_height && inw >= 0 && inw < in_width) {
                  index_t input_offset = in_base + inh * in_width + inw;
                  res = std::max(res, input[input_offset]);
                }
              }
            }
            output[out_offset] = res;
          }
        }
      }
    }
  }

  void AvgPooling(const float *input,
                  const index_t batch,
                  const index_t in_height,
                  const index_t in_width,
                  const index_t channels,
                  const index_t out_height,
                  const index_t out_width,
                  const int filter_height,
                  const int filter_width,
                  const int stride_h,
                  const int stride_w,
                  const int dilation_h,
                  const int dilation_w,
                  const int pad_top,
                  const int pad_left,
                  float *output) {
    const index_t in_image_size = in_height * in_width;
    const index_t out_image_size = out_height * out_width;
    const index_t in_batch_size = channels * in_image_size;
    const index_t out_batch_size = channels * out_image_size;

#pragma omp parallel for collapse(2)
    for (index_t b = 0; b < batch; ++b) {
      for (index_t c = 0; c < channels; ++c) {
        const index_t out_base = b * out_batch_size + c * out_image_size;
        const index_t in_base = b * in_batch_size + c * in_image_size;
        for (index_t h = 0; h < out_height; ++h) {
          for (index_t w = 0; w < out_width; ++w) {
            const index_t out_offset = out_base + h * out_width + w;
            float res = 0;
            int block_size = 0;
            for (int fh = 0; fh < filter_height; ++fh) {
              for (int fw = 0; fw < filter_width; ++fw) {
                int inh = h * stride_h + dilation_h * fh - pad_top;
                int inw = w * stride_w + dilation_w * fw - pad_left;
                if (inh >= 0 && inh < in_height && inw >= 0 && inw < in_width) {
                  index_t input_offset = in_base + inh * in_width + inw;
                  res += input[input_offset];
                  ++block_size;
                }
              }
            }
            output[out_offset] = res / block_size;
          }
        }
      }
    }
  }

  void operator()(const Tensor *input_tensor,
                  Tensor *output_tensor,
                  StatsFuture *future) {
    MACE_UNUSED(future);
    std::vector<index_t> output_shape(4);
    std::vector<index_t> filter_shape = {
      input_tensor->dim(1), input_tensor->dim(1), kernels_[0], kernels_[1]};

    std::vector<int> paddings(2);
    if (paddings_.empty()) {
      kernels::CalcNCHWPaddingAndOutputSize(
        input_tensor->shape().data(), filter_shape.data(), dilations_,
        strides_, padding_type_, output_shape.data(), paddings.data());
    } else {
      paddings = paddings_;
      CalcNCHWOutputSize(input_tensor->shape().data(),
                         filter_shape.data(),
                         paddings_.data(),
                         dilations_,
                         strides_,
                         RoundType::CEIL,
                         output_shape.data());
    }
    output_tensor->Resize(output_shape);

    Tensor::MappingGuard input_guard(input_tensor);
    Tensor::MappingGuard output_guard(output_tensor);
    const float *input = input_tensor->data<float>();
    float *output = output_tensor->mutable_data<float>();
    const index_t *input_shape = input_tensor->shape().data();
    index_t batch = output_shape[0];
    index_t channels = output_shape[1];
    index_t height = output_shape[2];
    index_t width = output_shape[3];

    index_t input_height = input_shape[2];
    index_t input_width = input_shape[3];

    int filter_h = kernels_[0];
    int filter_w = kernels_[1];

    int stride_h = strides_[0];
    int stride_w = strides_[1];

    int dilation_h = dilations_[0];
    int dilation_w = dilations_[1];

    int pad_top = paddings[0] / 2;
    int pad_left = paddings[1] / 2;

    if (pooling_type_ == PoolingType::MAX) {
      MaxPooling(input,
                 batch,
                 input_height,
                 input_width,
                 channels,
                 height,
                 width,
                 filter_h,
                 filter_w,
                 stride_h,
                 stride_w,
                 dilation_h,
                 dilation_w,
                 pad_top,
                 pad_left,
                 output);
    } else if (pooling_type_ == PoolingType::AVG) {
      AvgPooling(input,
                 batch,
                 input_height,
                 input_width,
                 channels,
                 height,
                 width,
                 filter_h,
                 filter_w,
                 stride_h,
                 stride_w,
                 dilation_h,
                 dilation_w,
                 pad_top,
                 pad_left,
                 output);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
struct PoolingFunctor<DeviceType::GPU, T> : PoolingFunctorBase {
  PoolingFunctor(const PoolingType pooling_type,
                 const int *kernels,
                 const int *strides,
                 const Padding padding_type,
                 const std::vector<int> &paddings,
                 const int *dilations)
      : PoolingFunctorBase(
            pooling_type, kernels, strides, padding_type, paddings, dilations) {
  }
  void operator()(const Tensor *input_tensor,
                  Tensor *output_tensor,
                  StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_POOLING_H_
