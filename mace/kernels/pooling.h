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
                  const index_t *in_shape,
                  const index_t *out_shape,
                  const int *filter_hw,
                  const int *stride_hw,
                  const int *dilation_hw,
                  const int *pad_hw,
                  float *output) {
    const index_t in_image_size = in_shape[2] * in_shape[3];
    const index_t out_image_size = out_shape[2] * out_shape[3];
    const index_t in_batch_size = in_shape[1] * in_image_size;
    const index_t out_batch_size = out_shape[1] * out_image_size;

#pragma omp parallel for collapse(2)
    for (index_t b = 0; b < out_shape[0]; ++b) {
      for (index_t c = 0; c < out_shape[1]; ++c) {
        const index_t out_base = b * out_batch_size + c * out_image_size;
        const index_t in_base = b * in_batch_size + c * in_image_size;
        const index_t out_height = out_shape[2];
        const index_t out_width = out_shape[3];
        const index_t in_height = in_shape[2];
        const index_t in_width = in_shape[3];

        for (index_t h = 0; h < out_height; ++h) {
          for (index_t w = 0; w < out_width; ++w) {
            const index_t out_offset = out_base + h * out_width + w;
            float res = std::numeric_limits<float>::lowest();
            for (int fh = 0; fh < filter_hw[0]; ++fh) {
              for (int fw = 0; fw < filter_hw[1]; ++fw) {
                index_t inh =
                    h * stride_hw[0] + dilation_hw[0] * fh - pad_hw[0];
                index_t inw =
                    w * stride_hw[1] + dilation_hw[1] * fw - pad_hw[1];
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
                  const index_t *in_shape,
                  const index_t *out_shape,
                  const int *filter_hw,
                  const int *stride_hw,
                  const int *dilation_hw,
                  const int *pad_hw,
                  float *output) {
    const index_t in_image_size = in_shape[2] * in_shape[3];
    const index_t out_image_size = out_shape[2] * out_shape[3];
    const index_t in_batch_size = in_shape[1] * in_image_size;
    const index_t out_batch_size = out_shape[1] * out_image_size;

#pragma omp parallel for collapse(2)
    for (index_t b = 0; b < out_shape[0]; ++b) {
      for (index_t c = 0; c < out_shape[1]; ++c) {
        const index_t out_base = b * out_batch_size + c * out_image_size;
        const index_t in_base = b * in_batch_size + c * in_image_size;
        const index_t in_height = in_shape[2];
        const index_t in_width = in_shape[3];
        const index_t out_height = out_shape[2];
        const index_t out_width = out_shape[3];
        for (index_t h = 0; h < out_height; ++h) {
          for (index_t w = 0; w < out_width; ++w) {
            const index_t out_offset = out_base + h * out_width + w;
            float res = 0;
            int block_size = 0;
            for (int fh = 0; fh < filter_hw[0]; ++fh) {
              for (int fw = 0; fw < filter_hw[1]; ++fw) {
                index_t inh =
                    h * stride_hw[0] + dilation_hw[0] * fh - pad_hw[0];
                index_t inw =
                    w * stride_hw[1] + dilation_hw[1] * fw - pad_hw[1];
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

  MaceStatus operator()(const Tensor *input_tensor,
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
    MACE_RETURN_IF_ERROR(output_tensor->Resize(output_shape));

    Tensor::MappingGuard input_guard(input_tensor);
    Tensor::MappingGuard output_guard(output_tensor);
    const float *input = input_tensor->data<float>();
    float *output = output_tensor->mutable_data<float>();
    const index_t *input_shape = input_tensor->shape().data();
    int pad_hw[2] = {paddings[0] / 2, paddings[1] / 2};

    if (pooling_type_ == PoolingType::MAX) {
      MaxPooling(input,
                 input_shape,
                 output_shape.data(),
                 kernels_,
                 strides_,
                 dilations_,
                 pad_hw,
                 output);
    } else if (pooling_type_ == PoolingType::AVG) {
      AvgPooling(input,
                 input_shape,
                 output_shape.data(),
                 kernels_,
                 strides_,
                 dilations_,
                 pad_hw,
                 output);
    } else {
      MACE_NOT_IMPLEMENTED;
    }

    return MACE_SUCCESS;
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
  MaceStatus operator()(const Tensor *input_tensor,
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
