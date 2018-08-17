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

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

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

  MaceStatus operator()(const Tensor *input_tensor,  // NCHW
                        Tensor *output_tensor,       // NCHW
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

template <>
struct PoolingFunctor<DeviceType::CPU, uint8_t>: PoolingFunctorBase {
  PoolingFunctor(const PoolingType pooling_type,
                 const int *kernels,
                 const int *strides,
                 const Padding padding_type,
                 const std::vector<int> &paddings,
                 const int *dilations)
      : PoolingFunctorBase(
      pooling_type, kernels, strides, padding_type, paddings, dilations) {
  }

  void MaxPooling(const uint8_t *input,
                  const index_t *in_shape,
                  const index_t *out_shape,
                  const int *filter_hw,
                  const int *stride_hw,
                  const int *pad_hw,
                  uint8_t *output) {
#pragma omp parallel for collapse(3)
    for (index_t b = 0; b < out_shape[0]; ++b) {
      for (index_t h = 0; h < out_shape[1]; ++h) {
        for (index_t w = 0; w < out_shape[2]; ++w) {
          const index_t out_height = out_shape[1];
          const index_t out_width = out_shape[2];
          const index_t channels = out_shape[3];
          const index_t in_height = in_shape[1];
          const index_t in_width = in_shape[2];
          const index_t in_h_base = h * stride_hw[0] - pad_hw[0];
          const index_t in_w_base = w * stride_hw[1] - pad_hw[1];
          const index_t in_h_begin = std::max<index_t>(0, in_h_base);
          const index_t in_w_begin = std::max<index_t>(0, in_w_base);
          const index_t in_h_end =
              std::min(in_height, in_h_base + filter_hw[0]);
          const index_t in_w_end =
              std::min(in_width, in_w_base + filter_hw[1]);

          uint8_t *out_ptr =
              output + ((b * out_height + h) * out_width + w) * channels;
          for (index_t ih = in_h_begin; ih < in_h_end; ++ih) {
            for (index_t iw = in_w_begin; iw < in_w_end; ++iw) {
              const uint8_t *in_ptr = input +
                  ((b * in_height + ih) * in_width + iw) * channels;
              index_t c = 0;
#if defined(MACE_ENABLE_NEON)
              for (; c <= channels - 16; c += 16) {
                uint8x16_t out_vec = vld1q_u8(out_ptr + c);
                uint8x16_t in_vec = vld1q_u8(in_ptr + c);
                out_vec = vmaxq_u8(out_vec, in_vec);
                vst1q_u8(out_ptr + c, out_vec);
              }
              for (; c <= channels - 8; c += 8) {
                uint8x8_t out_vec = vld1_u8(out_ptr + c);
                uint8x8_t in_vec = vld1_u8(in_ptr + c);
                out_vec = vmax_u8(out_vec, in_vec);
                vst1_u8(out_ptr + c, out_vec);
              }
#endif
              for (; c < channels; ++c) {
                out_ptr[c] = std::max(out_ptr[c], in_ptr[c]);
              }
            }
          }
        }
      }
    }
  }

  void AvgPooling(const uint8_t *input,
                  const index_t *in_shape,
                  const index_t *out_shape,
                  const int *filter_hw,
                  const int *stride_hw,
                  const int *pad_hw,
                  uint8_t *output) {
#pragma omp parallel for collapse(3)
    for (index_t b = 0; b < out_shape[0]; ++b) {
      for (index_t h = 0; h < out_shape[1]; ++h) {
        for (index_t w = 0; w < out_shape[2]; ++w) {
          const index_t out_height = out_shape[1];
          const index_t out_width = out_shape[2];
          const index_t channels = out_shape[3];
          const index_t in_height = in_shape[1];
          const index_t in_width = in_shape[2];
          const index_t in_h_base = h * stride_hw[0] - pad_hw[0];
          const index_t in_w_base = w * stride_hw[1] - pad_hw[1];
          const index_t in_h_begin = std::max<index_t>(0, in_h_base);
          const index_t in_w_begin = std::max<index_t>(0, in_w_base);
          const index_t in_h_end =
              std::min(in_height, in_h_base + filter_hw[0]);
          const index_t in_w_end =
              std::min(in_width, in_w_base + filter_hw[1]);
          const index_t block_size =
              (in_h_end - in_h_begin) * (in_w_end - in_w_begin);
          MACE_CHECK(block_size > 0);

          std::vector<uint16_t> average_buffer(channels);
          uint16_t *avg_buffer = average_buffer.data();
          std::fill_n(avg_buffer, channels, 0);
          for (index_t ih = in_h_begin; ih < in_h_end; ++ih) {
            for (index_t iw = in_w_begin; iw < in_w_end; ++iw) {
              const uint8_t *in_ptr = input +
                  ((b * in_height + ih) * in_width + iw) * channels;
              index_t c = 0;
#if defined(MACE_ENABLE_NEON)
              for (; c <= channels - 16; c += 16) {
                uint16x8_t avg_vec[2];
                avg_vec[0] = vld1q_u16(avg_buffer + c);
                avg_vec[1] = vld1q_u16(avg_buffer + c + 8);
                uint8x16_t in_vec = vld1q_u8(in_ptr + c);
                avg_vec[0] = vaddw_u8(avg_vec[0], vget_low_u8(in_vec));
                avg_vec[1] = vaddw_u8(avg_vec[1], vget_high_u8(in_vec));
                vst1q_u16(avg_buffer + c, avg_vec[0]);
                vst1q_u16(avg_buffer + c + 8, avg_vec[1]);
              }
              for (; c <= channels - 8; c += 8) {
                uint16x8_t avg_vec = vld1q_u16(avg_buffer + c);
                uint8x8_t in_vec = vld1_u8(in_ptr + c);
                avg_vec = vaddw_u8(avg_vec, in_vec);
                vst1q_u16(avg_buffer + c, avg_vec);
              }
#endif
              for (; c < channels; ++c) {
                avg_buffer[c] += in_ptr[c];
              }
            }
          }
          uint8_t *out_ptr =
              output + ((b * out_height + h) * out_width + w) * channels;
          for (index_t c = 0; c < channels; ++c) {
            out_ptr[c] = static_cast<uint8_t>(
                (avg_buffer[c] + block_size / 2) / block_size);
          }
        }
      }
    }
  }

  MaceStatus operator()(const Tensor *input_tensor,  // NHWC
                        Tensor *output_tensor,       // NHWC
                        StatsFuture *future) {
    MACE_UNUSED(future);
    MACE_CHECK(dilations_[0] == 1 && dilations_[1] == 1,
               "Quantized pooling does not support dilation > 1 yet.");
    // Use the same scale and zero point with input and output.
    output_tensor->SetScale(input_tensor->scale());
    output_tensor->SetZeroPoint(input_tensor->zero_point());

    std::vector<index_t> output_shape(4);
    std::vector<index_t> filter_shape = {
        input_tensor->dim(3), kernels_[0], kernels_[1], input_tensor->dim(3)};

    std::vector<int> paddings(2);
    if (paddings_.empty()) {
      CalcPaddingAndOutputSize(input_tensor->shape().data(),
                               NHWC,
                               filter_shape.data(),
                               OHWI,
                               dilations_,
                               strides_,
                               padding_type_,
                               output_shape.data(),
                               paddings.data());
    } else {
      paddings = paddings_;
      CalcOutputSize(input_tensor->shape().data(),
                     NHWC,
                     filter_shape.data(),
                     OHWI,
                     paddings_.data(),
                     dilations_,
                     strides_,
                     RoundType::CEIL,
                     output_shape.data());
    }
    MACE_RETURN_IF_ERROR(output_tensor->Resize(output_shape));

    const index_t out_channels = output_tensor->dim(3);
    const index_t in_channels = input_tensor->dim(3);
    MACE_CHECK(out_channels == in_channels);

    Tensor::MappingGuard input_guard(input_tensor);
    Tensor::MappingGuard output_guard(output_tensor);
    const uint8_t *input = input_tensor->data<uint8_t>();
    uint8_t *output = output_tensor->mutable_data<uint8_t>();
    int pad_hw[2] = {paddings[0] / 2, paddings[1] / 2};

    if (pooling_type_ == PoolingType::MAX) {
      MaxPooling(input,
                 input_tensor->shape().data(),
                 output_shape.data(),
                 kernels_,
                 strides_,
                 pad_hw,
                 output);
    } else if (pooling_type_ == PoolingType::AVG) {
      AvgPooling(input,
                 input_tensor->shape().data(),
                 output_shape.data(),
                 kernels_,
                 strides_,
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
