// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/arm/base/conv_2d_general.h"

#include <memory>

#include "mace/ops/arm/base/common_neon.h"

namespace mace {
namespace ops {
namespace arm {

template<typename T>
MaceStatus Conv2dGeneral<T>::Compute(const OpContext *context,
                                     const Tensor *input,
                                     const Tensor *filter,
                                     Tensor *output) {
  std::unique_ptr<const Tensor> padded_input;
  std::unique_ptr<Tensor> padded_output;
  ResizeOutAndPadInOut(context, input, filter, output, 1, 4,
                       &padded_input, &padded_output);
  const Tensor *in_tensor = input;
  if (padded_input != nullptr) {
    in_tensor = padded_input.get();
  }
  Tensor *out_tensor = output;
  if (padded_output != nullptr) {
    out_tensor = padded_output.get();
  }
  out_tensor->Clear();

  Tensor::MappingGuard in_guard(input);
  Tensor::MappingGuard filter_guard(filter);
  Tensor::MappingGuard out_guard(output);

  const T *filter_data = filter->data<T>();
  const T *input_data = in_tensor->data<T>();
  T *output_data = out_tensor->mutable_data<T>();

  const ConvComputeParam p =
      PreWorkAndGetConv2DParam(context, in_tensor, out_tensor);
  auto &filter_shape = filter->shape();

  DoCompute(p, filter_data, input_data, output_data, filter_shape);

  UnPadOutput(*out_tensor, output);
  return MaceStatus::MACE_SUCCESS;
}

template<typename T>
MaceStatus Conv2dGeneral<T>::DoCompute(
    const ConvComputeParam &p, const T *filter_data,
    const T *input_data, T *output_data,
    const std::vector<index_t> &filter_shape) {
  const index_t filter_height = filter_shape[2];
  const index_t filter_width = filter_shape[3];
  const index_t filter_size = filter_height * filter_width;

  p.thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        const int stride_h = strides_[0];
        const int stride_w = strides_[1];
        const int dilation_h = dilations_[0];
        const int dilation_w = dilations_[1];
        if (m + 3 < p.out_channels) {
          T *out_ptr0_base =
              output_data + b * p.out_batch_size + m * p.out_image_size;
          T *out_ptr1_base = out_ptr0_base + p.out_image_size;
          T *out_ptr2_base = out_ptr1_base + p.out_image_size;
          T *out_ptr3_base = out_ptr2_base + p.out_image_size;
          for (index_t h = 0; h < p.out_height; ++h) {
            const index_t ih = h * stride_h;
            for (index_t w = 0; w + 3 < p.out_width; w += 4) {
              const index_t iw = w * stride_w;
              index_t out_offset = h * p.out_width + w;
              float32x4_t vo0 = vdupq_n_f32(0.f);
              float32x4_t vo1 = vdupq_n_f32(0.f);
              float32x4_t vo2 = vdupq_n_f32(0.f);
              float32x4_t vo3 = vdupq_n_f32(0.f);
              const T *in_ptr_base = input_data + b * p.in_batch_size;
              const T *filter_ptr0 =
                  filter_data + m * p.in_channels * filter_size;
              const T *filter_ptr1 = filter_ptr0 + p.in_channels * filter_size;
              const T *filter_ptr2 = filter_ptr1 + p.in_channels * filter_size;
              const T *filter_ptr3 = filter_ptr2 + p.in_channels * filter_size;
              for (index_t c = 0; c < p.in_channels; ++c) {
                index_t in_offset = ih * p.in_width + iw;
                // calc by row
                for (index_t kh = 0; kh < filter_height; ++kh) {
                  for (index_t kw = 0; kw < filter_width; ++kw) {
                    const T i0 = in_ptr_base[in_offset + kw * dilation_w];
                    const T i1 =
                        in_ptr_base[in_offset + stride_w + kw * dilation_w];
                    const T i2 =
                        in_ptr_base[in_offset + 2 * stride_w + kw * dilation_w];
                    const T i3 =
                        in_ptr_base[in_offset + 3 * stride_w + kw * dilation_w];
                    const T f0 = filter_ptr0[kw];
                    const T f1 = filter_ptr1[kw];
                    const T f2 = filter_ptr2[kw];
                    const T f3 = filter_ptr3[kw];
                    // outch 0
                    vo0[0] += i0 * f0;
                    vo0[1] += i1 * f0;
                    vo0[2] += i2 * f0;
                    vo0[3] += i3 * f0;
                    // outch 1
                    vo1[0] += i0 * f1;
                    vo1[1] += i1 * f1;
                    vo1[2] += i2 * f1;
                    vo1[3] += i3 * f1;
                    // outch 2
                    vo2[0] += i0 * f2;
                    vo2[1] += i1 * f2;
                    vo2[2] += i2 * f2;
                    vo2[3] += i3 * f2;
                    // outch 3
                    vo3[0] += i0 * f3;
                    vo3[1] += i1 * f3;
                    vo3[2] += i2 * f3;
                    vo3[3] += i3 * f3;
                  }  // kw

                  in_offset += dilation_h * p.in_width;
                  filter_ptr0 += filter_width;
                  filter_ptr1 += filter_width;
                  filter_ptr2 += filter_width;
                  filter_ptr3 += filter_width;
                }  // kh
                in_ptr_base += p.in_image_size;
              }  // c
              vst1q(out_ptr0_base + out_offset, vo0);
              vst1q(out_ptr1_base + out_offset, vo1);
              vst1q(out_ptr2_base + out_offset, vo2);
              vst1q(out_ptr3_base + out_offset, vo3);
            }  // w
          }  // h
        } else {
          for (index_t mm = m; mm < p.out_channels; ++mm) {
            T *out_ptr0_base =
                output_data + b * p.out_batch_size + mm * p.out_image_size;
            for (index_t h = 0; h < p.out_height; ++h) {
              for (index_t w = 0; w + 3 < p.out_width; w += 4) {
                // input offset
                const index_t ih = h * stride_h;
                const index_t iw = w * stride_w;
                // output offset
                const index_t out_offset = h * p.out_width + w;
                // output (1 outch x 1 height x 4 width): vo_outch_height
                float32x4_t vo0 = vdupq_n_f32(0.f);
                const T *in_ptr_base = input_data + b * p.in_batch_size;
                const T *filter_ptr0 =
                    filter_data + mm * p.in_channels * filter_size;
                for (index_t c = 0; c < p.in_channels; ++c) {
                  index_t in_offset = ih * p.in_width + iw;
                  for (index_t kh = 0; kh < filter_height; ++kh) {
                    for (index_t kw = 0; kw < filter_width; ++kw) {
                      T i0 = in_ptr_base[in_offset + kw * dilation_w];
                      T i1 = in_ptr_base[in_offset + stride_w +
                          kw * dilation_w];
                      T i2 = in_ptr_base[in_offset + 2 * stride_w +
                          kw * dilation_w];
                      T i3 = in_ptr_base[in_offset + 3 * stride_w +
                          kw * dilation_w];
                      T f0 = filter_ptr0[kw];
                      // outch 0
                      vo0[0] += i0 * f0;
                      vo0[1] += i1 * f0;
                      vo0[2] += i2 * f0;
                      vo0[3] += i3 * f0;
                    }  // kw
                    in_offset += dilation_h * p.in_width;
                    filter_ptr0 += filter_width;
                  }  // kh
                  in_ptr_base += p.in_image_size;
                }  // c
                vst1q(out_ptr0_base + out_offset, vo0);
              }  // w
            }  // h
          }  // mm
        }  // if
      }  // m
    }  // b
  }, 0, p.batch, 1, 0, p.out_channels, 4);

  return MaceStatus::MACE_SUCCESS;
}

void RegisterConv2dGeneralDelegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, Conv2dGeneral<float>, delegator::Conv2dParam,
      MACE_DELEGATOR_KEY(Conv2d, DeviceType::CPU, float, ImplType::NEON));

  MACE_REGISTER_BF16_DELEGATOR(
      registry, Conv2dGeneral<BFloat16>, delegator::Conv2dParam,
      MACE_DELEGATOR_KEY(Conv2d, DeviceType::CPU, BFloat16, ImplType::NEON));
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
