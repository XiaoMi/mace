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

#ifndef MACE_KERNELS_CONV_2D_H_
#define MACE_KERNELS_CONV_2D_H_

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <algorithm>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/tensor.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <typename T,
          int inc_tile_size,
          int c_count,
          int h_count,
          int w_count>
void Conv2dKernelFunc(const T *input_ptr,  // batch start
                      const T *filter_ptr,
                      const T *bias_ptr,
                      T *output_ptr,  // batch start
                      const int h_offset,
                      const int w_offset,
                      const int c_offset,
                      const int kernel_h,
                      const int kernel_w,
                      const int stride_h,
                      const int stride_w,
                      const int dilation_h,
                      const int dilation_w,
                      const int channels,
                      const int input_channels,
                      const int width,
                      const int padded_width) {
  T sum[h_count * w_count * c_count] = {0.0f};
  if (bias_ptr != nullptr) {
    for (int hi = 0; hi < h_count; ++hi) {
      for (int wi = 0; wi < w_count; ++wi) {
        for (int ci = 0; ci < c_count; ++ci) {
          const int sum_idx = (hi * w_count + wi) * c_count + ci;
          sum[sum_idx] = bias_ptr[c_offset + ci];
        }
      }
    }
  }

  for (int kh = 0; kh < kernel_h; ++kh) {
    for (int kw = 0; kw < kernel_w; ++kw) {
      int inc = 0;
      for (; inc + inc_tile_size <= input_channels; inc += inc_tile_size) {
#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
        // AArch64 NEON has 32 128-bit general purpose registers
        static_assert(inc_tile_size == 4, "input channels tile size must be 4");
        float32x4_t in[h_count * w_count];  // NOLINT(runtime/arrays)
#else
        T in[h_count * w_count * inc_tile_size];  // NOLINT(runtime/arrays)
#endif
        for (int hi = 0; hi < h_count; ++hi) {
          for (int wi = 0; wi < w_count; ++wi) {
            const int in_idx = hi * w_count + wi;
            const int inh = (h_offset + hi) * stride_h + kh * dilation_h;
            const int inw = (w_offset + wi) * stride_w + kw * dilation_w;
            const int in_offset =
                (inh * padded_width + inw) * input_channels + inc;
#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
            static_assert(inc_tile_size == 4,
                          "input channels tile size must be 4");
            in[in_idx] = vld1q_f32(input_ptr + in_offset);
#else
            for (int inci = 0; inci < inc_tile_size; ++inci) {
              in[in_idx * inc_tile_size + inci] = input_ptr[in_offset + inci];
            }
#endif
          }
        }

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
        static_assert(inc_tile_size == 4, "input channels tile size must be 4");
        float32x4_t weights[c_count];  // NOLINT(runtime/arrays)
#else
        T weights[c_count * inc_tile_size];  // NOLINT(runtime/arrays)
#endif
        for (int ci = 0; ci < c_count; ++ci) {
          const int weights_idx = ci;
          const int filter_offset =
              ((kh * kernel_w + kw) * channels + c_offset + ci) *
                  input_channels +
              inc;
#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
          weights[weights_idx] = vld1q_f32(filter_ptr + filter_offset);
#else
          for (int inci = 0; inci < inc_tile_size; ++inci) {
            weights[weights_idx * inc_tile_size + inci] =
                filter_ptr[filter_offset + inci];
          }
#endif
        }
        for (int hi = 0; hi < h_count; ++hi) {
          for (int wi = 0; wi < w_count; ++wi) {
            for (int ci = 0; ci < c_count; ++ci) {
              const int weights_idx = ci;
              const int in_idx = hi * w_count + wi;
              const int sum_idx = (hi * w_count + wi) * c_count + ci;
#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
              float32x4_t tmp = vmulq_f32(in[in_idx], weights[weights_idx]);
              sum[sum_idx] += vaddvq_f32(tmp);
#else
              for (int inci = 0; inci < inc_tile_size; ++inci) {
                sum[sum_idx] += in[in_idx * inc_tile_size + inci] *
                                weights[weights_idx * inc_tile_size + inci];
              }
#endif
            }
          }
        }
      }
      // handling the remaining input channels
      for (; inc < input_channels; ++inc) {
        T in[h_count * w_count];  // NOLINT(runtime/arrays)
        for (int hi = 0; hi < h_count; ++hi) {
          for (int wi = 0; wi < w_count; ++wi) {
            const int in_idx = hi * w_count + wi;
            const int inh = (h_offset + hi) * stride_h + kh * dilation_h;
            const int inw = (w_offset + wi) * stride_w + kw * dilation_w;
            const int in_offset =
                (inh * padded_width + inw) * input_channels + inc;
            in[in_idx] = input_ptr[in_offset];
          }
        }

        T weights[c_count];  // NOLINT(runtime/arrays)
        for (int ci = 0; ci < c_count; ++ci) {
          const int weights_idx = ci;
          const int filter_offset =
              ((kh * kernel_w + kw) * channels + c_offset + ci) *
                  input_channels +
              inc;
          weights[weights_idx] = filter_ptr[filter_offset];
        }
        for (int hi = 0; hi < h_count; ++hi) {
          for (int wi = 0; wi < w_count; ++wi) {
            for (int ci = 0; ci < c_count; ++ci) {
              const int weights_idx = ci;
              const int in_idx = hi * w_count + wi;
              const int sum_idx = (hi * w_count + wi) * c_count + ci;
              sum[sum_idx] += in[in_idx] * weights[weights_idx];
            }
          }
        }
      }
    }
  }
  // save output
  for (int hi = 0; hi < h_count; ++hi) {
    for (int wi = 0; wi < w_count; ++wi) {
      for (int ci = 0; ci < c_count; ++ci) {
        const int out_offset =
            ((h_offset + hi) * width + w_offset + wi) * channels + c_offset +
            ci;
        const int sum_idx = (hi * w_count + wi) * c_count + ci;
        output_ptr[out_offset] = sum[sum_idx];
      }
    }
  }
}

struct Conv2dFunctorBase {
  Conv2dFunctorBase(const int *strides,
                    const Padding &padding_type,
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

#define MACE_DO_CONV2D(CC, CH, CW) \
Conv2dKernelFunc<T, inc_tile_size, CC, CH, CW>( \
  input_ptr, filter_data, bias_data, output_ptr, \
  h_offset, w_offset, c_offset, kernel_h, kernel_w, \
  stride_h, stride_w, dilation_h, dilation_w, \
  channels, input_channels, width, padded_width);

#define MACE_CASE_W_CONV2D(CC, CH) \
switch (w_count) { \
  case 1: \
    MACE_DO_CONV2D(CC, CH, 1); \
    break; \
  case 2: \
    MACE_DO_CONV2D(CC, CH, 2); \
    break; \
  case 3: \
    MACE_DO_CONV2D(CC, CH, 3); \
    break; \
  case 4: \
    MACE_DO_CONV2D(CC, CH, 4); \
    break; \
  default: \
    LOG(FATAL) << "Unsupported w tile: " << w_count; \
}

#define MACE_CASE_H_CONV2D(CC) \
switch (h_count) { \
  case 1: \
    MACE_CASE_W_CONV2D(CC, 1); \
    break; \
  case 2: \
    MACE_CASE_W_CONV2D(CC, 2); \
    break; \
  default: \
    LOG(FATAL) << "Unsupported h tile: " << h_count; \
}

#define MACE_CASE_C_CONV2D \
switch (c_count) { \
  case 1: \
    MACE_CASE_H_CONV2D(1); \
    break; \
  case 2: \
    MACE_CASE_H_CONV2D(2); \
    break; \
  case 3: \
    MACE_CASE_H_CONV2D(3); \
    break; \
  case 4: \
    MACE_CASE_H_CONV2D(4); \
    break; \
  case 5: \
    MACE_CASE_H_CONV2D(5); \
    break; \
  case 6: \
    MACE_CASE_H_CONV2D(6); \
    break; \
  case 7: \
    MACE_CASE_H_CONV2D(7); \
    break; \
  case 8: \
    MACE_CASE_H_CONV2D(8); \
    break; \
  case 9: \
    MACE_CASE_H_CONV2D(9); \
    break; \
  case 10: \
    MACE_CASE_H_CONV2D(10); \
    break; \
  case 11: \
    MACE_CASE_H_CONV2D(11); \
    break; \
  case 12: \
    MACE_CASE_H_CONV2D(12); \
    break; \
  case 13: \
    MACE_CASE_H_CONV2D(13); \
    break; \
  case 14: \
    MACE_CASE_H_CONV2D(14); \
    break; \
  case 15: \
    MACE_CASE_H_CONV2D(15); \
    break; \
  case 16: \
    MACE_CASE_H_CONV2D(16); \
    break; \
  default: \
    LOG(FATAL) << "Unsupported c tile: " << c_count; \
}


template <DeviceType D, typename T>
struct Conv2dFunctor : Conv2dFunctorBase {
  Conv2dFunctor(const int *strides,
                const Padding &padding_type,
                const std::vector<int> &paddings,
                const int *dilations,
                const ActivationType activation,
                const float relux_max_limit,
                ScratchBuffer *scratch)
      : Conv2dFunctorBase(strides,
                          padding_type,
                          paddings,
                          dilations,
                          activation,
                          relux_max_limit) {}

  void operator()(const Tensor *input,   // NHWC
                  const Tensor *filter,  // HWOI
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_CHECK_NOTNULL(input);
    MACE_CHECK_NOTNULL(filter);
    MACE_CHECK_NOTNULL(output);

    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);
    if (paddings_.empty()) {
      kernels::CalcNHWCPaddingAndOutputSize(
          input->shape().data(), filter->shape().data(), dilations_, strides_,
          padding_type_, output_shape.data(), paddings.data());
    } else {
      paddings = paddings_;
      CalcOutputSize(input->shape().data(), filter->shape().data(),
                     paddings_.data(), dilations_, strides_, RoundType::FLOOR,
                     output_shape.data());
    }
    output->Resize(output_shape);

    int batch = output->dim(0);
    int height = output->dim(1);
    int width = output->dim(2);
    int channels = output->dim(3);

    int input_batch = input->dim(0);
    int input_height = input->dim(1);
    int input_width = input->dim(2);
    int input_channels = input->dim(3);

    int kernel_h = filter->dim(0);
    int kernel_w = filter->dim(1);
    MACE_CHECK(filter->dim(2) == channels, filter->dim(2), " != ", channels);
    MACE_CHECK(filter->dim(3) == input_channels, filter->dim(3), " != ",
               input_channels);

    int stride_h = strides_[0];
    int stride_w = strides_[1];

    int dilation_h = dilations_[0];
    int dilation_w = dilations_[1];

    MACE_CHECK(batch == input_batch, "Input/Output batch size mismatch");

    int padded_height = input_height + paddings[0];
    int padded_width = input_width + paddings[1];

    Tensor padded_input;
    // Keep this alive during kernel execution
    if (paddings[0] > 0 || paddings[1] > 0) {
      ConstructNHWCInputWithPadding(input, paddings.data(), &padded_input);
      input = &padded_input;
    }

    // padded_input.DebugPrint();

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard filter_mapper(filter);
    Tensor::MappingGuard bias_mapper(bias);
    Tensor::MappingGuard output_mapper(output);
    auto input_data = input->data<T>();
    auto filter_data = filter->data<T>();
    auto bias_data = bias == nullptr ? nullptr : bias->data<T>();
    auto output_data = output->mutable_data<T>();

    constexpr int inc_tile_size = 4;
// TODO(heliangliang) Auto tuning these parameters
#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
    const int c_tile_size = 4;
    const int h_tile_size = 2;
    const int w_tile_size = 2;
#else
    const int c_tile_size = 4;
    const int h_tile_size = 1;
    const int w_tile_size = 2;
#endif

    const int c_tiles = RoundUpDiv(channels, c_tile_size);
    const int h_tiles = RoundUpDiv(height, h_tile_size);
    const int w_tiles = RoundUpDiv(width, w_tile_size);

#pragma omp parallel for collapse(4)
    for (int n = 0; n < batch; ++n) {
      for (int cb = 0; cb < c_tiles; ++cb) {
        for (int hb = 0; hb < h_tiles; ++hb) {
          for (int wb = 0; wb < w_tiles; ++wb) {
            const T *input_ptr =
                input_data + n * padded_height * padded_width * input_channels;
            T *output_ptr = output_data + n * height * width * channels;
            const int h_offset = hb * h_tile_size;
            const int w_offset = wb * w_tile_size;
            const int c_offset = cb * c_tile_size;

            const int h_count = std::min(h_tile_size, height - h_offset);
            const int w_count = std::min(w_tile_size, width - w_offset);
            const int c_count = std::min(c_tile_size, channels - c_offset);

            MACE_CASE_C_CONV2D;
          }
        }
      }
    }
    DoActivation(output_data, output_data, output->size(), activation_,
                 relux_max_limit_);
  }
};

template <>
struct Conv2dFunctor<DeviceType::NEON, float> : Conv2dFunctorBase {
  Conv2dFunctor(const int *strides,
                const Padding &padding_type,
                const std::vector<int> &paddings,
                const int *dilations,
                const ActivationType activation,
                const float relux_max_limit,
                ScratchBuffer *scratch)
    : Conv2dFunctorBase(strides,
                        padding_type,
                        paddings,
                        dilations,
                        activation,
                        relux_max_limit),
      is_filter_transformed_(false),
      scratch_(scratch) {}

  void operator()(const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future);

  Tensor transformed_filter_;
  bool is_filter_transformed_;
  ScratchBuffer *scratch_;
};

template <typename T>
struct Conv2dFunctor<DeviceType::OPENCL, T> : Conv2dFunctorBase {
  Conv2dFunctor(const int *strides,
                const Padding &padding_type,
                const std::vector<int> &paddings,
                const int *dilations,
                const ActivationType activation,
                const float relux_max_limit,
                ScratchBuffer *scratch)
      : Conv2dFunctorBase(strides,
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

#endif  // MACE_KERNELS_CONV_2D_H_
