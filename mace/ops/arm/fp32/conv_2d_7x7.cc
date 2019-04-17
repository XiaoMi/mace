// Copyright 2019 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/arm/fp32/conv_2d_7x7.h"

#include <arm_neon.h>
#include <memory>

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

#define MACE_Conv2dArmv8NeonK7x7SnLoadCalc4        \
  /* load filter (4 outch x 1 height x 4 width) */ \
  float32x4_t vf00, vf01;                          \
  float32x4_t vf10, vf11;                          \
  float32x4_t vf20, vf21;                          \
  float32x4_t vf30, vf31;                          \
  vf00 = vld1q_f32(filter_ptr0);                   \
  vf01 = vld1q_f32(filter_ptr0 + 3);               \
  vf10 = vld1q_f32(filter_ptr1);                   \
  vf11 = vld1q_f32(filter_ptr1 + 3);               \
  vf20 = vld1q_f32(filter_ptr2);                   \
  vf21 = vld1q_f32(filter_ptr2 + 3);               \
  vf30 = vld1q_f32(filter_ptr3);                   \
  vf31 = vld1q_f32(filter_ptr3 + 3);               \
                                                   \
  /* outch 0 */                                    \
  vo0 = vfmaq_laneq_f32(vo0, vi0, vf00, 0);        \
  vo0 = vfmaq_laneq_f32(vo0, vi1, vf00, 1);        \
  vo0 = vfmaq_laneq_f32(vo0, vi2, vf00, 2);        \
  vo0 = vfmaq_laneq_f32(vo0, vi3, vf00, 3);        \
  vo0 = vfmaq_laneq_f32(vo0, vi4, vf01, 1);        \
  vo0 = vfmaq_laneq_f32(vo0, vi5, vf01, 2);        \
  vo0 = vfmaq_laneq_f32(vo0, vi6, vf01, 3);        \
                                                   \
  /* outch 1 */                                    \
  vo1 = vfmaq_laneq_f32(vo1, vi0, vf10, 0);        \
  vo1 = vfmaq_laneq_f32(vo1, vi1, vf10, 1);        \
  vo1 = vfmaq_laneq_f32(vo1, vi2, vf10, 2);        \
  vo1 = vfmaq_laneq_f32(vo1, vi3, vf10, 3);        \
  vo1 = vfmaq_laneq_f32(vo1, vi4, vf11, 1);        \
  vo1 = vfmaq_laneq_f32(vo1, vi5, vf11, 2);        \
  vo1 = vfmaq_laneq_f32(vo1, vi6, vf11, 3);        \
                                                   \
  /* outch 2 */                                    \
  vo2 = vfmaq_laneq_f32(vo2, vi0, vf20, 0);        \
  vo2 = vfmaq_laneq_f32(vo2, vi1, vf20, 1);        \
  vo2 = vfmaq_laneq_f32(vo2, vi2, vf20, 2);        \
  vo2 = vfmaq_laneq_f32(vo2, vi3, vf20, 3);        \
  vo2 = vfmaq_laneq_f32(vo2, vi4, vf21, 1);        \
  vo2 = vfmaq_laneq_f32(vo2, vi5, vf21, 2);        \
  vo2 = vfmaq_laneq_f32(vo2, vi6, vf21, 3);        \
                                                   \
  /* outch 3 */                                    \
  vo3 = vfmaq_laneq_f32(vo3, vi0, vf30, 0);        \
  vo3 = vfmaq_laneq_f32(vo3, vi1, vf30, 1);        \
  vo3 = vfmaq_laneq_f32(vo3, vi2, vf30, 2);        \
  vo3 = vfmaq_laneq_f32(vo3, vi3, vf30, 3);        \
  vo3 = vfmaq_laneq_f32(vo3, vi4, vf31, 1);        \
  vo3 = vfmaq_laneq_f32(vo3, vi5, vf31, 2);        \
  vo3 = vfmaq_laneq_f32(vo3, vi6, vf31, 3);

#define MACE_Conv2dArmv8NeonK7x7SnLoadCalc1        \
  /* load filter (1 outch x 1 height x 4 width) */ \
  float32x4_t vf00, vf01;                          \
  vf00 = vld1q_f32(filter_ptr0);                   \
  vf01 = vld1q_f32(filter_ptr0 + 3);               \
                                                   \
  /* outch 0 */                                    \
  vo0 = vfmaq_laneq_f32(vo0, vi0, vf00, 0);        \
  vo0 = vfmaq_laneq_f32(vo0, vi1, vf00, 1);        \
  vo0 = vfmaq_laneq_f32(vo0, vi2, vf00, 2);        \
  vo0 = vfmaq_laneq_f32(vo0, vi3, vf00, 3);        \
  vo0 = vfmaq_laneq_f32(vo0, vi4, vf01, 1);        \
  vo0 = vfmaq_laneq_f32(vo0, vi5, vf01, 2);        \
  vo0 = vfmaq_laneq_f32(vo0, vi6, vf01, 3);

#define MACE_Conv2dArmv7NeonK7x7SnLoadCalc4               \
  /* load filter (4 outch x 1 height x 4 width) */        \
  float32x4_t vf00, vf01;                                 \
  float32x4_t vf10, vf11;                                 \
  float32x4_t vf20, vf21;                                 \
  float32x4_t vf30, vf31;                                 \
  vf00 = vld1q_f32(filter_ptr0);                          \
  vf01 = vld1q_f32(filter_ptr0 + 3);                      \
  vf10 = vld1q_f32(filter_ptr1);                          \
  vf11 = vld1q_f32(filter_ptr1 + 3);                      \
  vf20 = vld1q_f32(filter_ptr2);                          \
  vf21 = vld1q_f32(filter_ptr2 + 3);                      \
  vf30 = vld1q_f32(filter_ptr3);                          \
  vf31 = vld1q_f32(filter_ptr3 + 3);                      \
                                                          \
  /* outch 0 */                                           \
  vo0 = vmlaq_lane_f32(vo0, vi0, vget_low_f32(vf00), 0);  \
  vo0 = vmlaq_lane_f32(vo0, vi1, vget_low_f32(vf00), 1);  \
  vo0 = vmlaq_lane_f32(vo0, vi2, vget_high_f32(vf00), 0); \
  vo0 = vmlaq_lane_f32(vo0, vi3, vget_high_f32(vf00), 1); \
  vo0 = vmlaq_lane_f32(vo0, vi4, vget_low_f32(vf01), 1);  \
  vo0 = vmlaq_lane_f32(vo0, vi5, vget_high_f32(vf01), 0); \
  vo0 = vmlaq_lane_f32(vo0, vi6, vget_high_f32(vf01), 1); \
                                                          \
  /* outch 1 */                                           \
  vo1 = vmlaq_lane_f32(vo1, vi0, vget_low_f32(vf10), 0);  \
  vo1 = vmlaq_lane_f32(vo1, vi1, vget_low_f32(vf10), 1);  \
  vo1 = vmlaq_lane_f32(vo1, vi2, vget_high_f32(vf10), 0); \
  vo1 = vmlaq_lane_f32(vo1, vi3, vget_high_f32(vf10), 1); \
  vo1 = vmlaq_lane_f32(vo1, vi4, vget_low_f32(vf11), 1);  \
  vo1 = vmlaq_lane_f32(vo1, vi5, vget_high_f32(vf11), 0); \
  vo1 = vmlaq_lane_f32(vo1, vi6, vget_high_f32(vf11), 1); \
                                                          \
  /* outch 2 */                                           \
  vo2 = vmlaq_lane_f32(vo2, vi0, vget_low_f32(vf20), 0);  \
  vo2 = vmlaq_lane_f32(vo2, vi1, vget_low_f32(vf20), 1);  \
  vo2 = vmlaq_lane_f32(vo2, vi2, vget_high_f32(vf20), 0); \
  vo2 = vmlaq_lane_f32(vo2, vi3, vget_high_f32(vf20), 1); \
  vo2 = vmlaq_lane_f32(vo2, vi4, vget_low_f32(vf21), 1);  \
  vo2 = vmlaq_lane_f32(vo2, vi5, vget_high_f32(vf21), 0); \
  vo2 = vmlaq_lane_f32(vo2, vi6, vget_high_f32(vf21), 1); \
                                                          \
  /* outch 3 */                                           \
  vo3 = vmlaq_lane_f32(vo3, vi0, vget_low_f32(vf30), 0);  \
  vo3 = vmlaq_lane_f32(vo3, vi1, vget_low_f32(vf30), 1);  \
  vo3 = vmlaq_lane_f32(vo3, vi2, vget_high_f32(vf30), 0); \
  vo3 = vmlaq_lane_f32(vo3, vi3, vget_high_f32(vf30), 1); \
  vo3 = vmlaq_lane_f32(vo3, vi4, vget_low_f32(vf31), 1);  \
  vo3 = vmlaq_lane_f32(vo3, vi5, vget_high_f32(vf31), 0); \
  vo3 = vmlaq_lane_f32(vo3, vi6, vget_high_f32(vf31), 1);

#define MACE_Conv2dArmv7NeonK7x7SnLoadCalc1               \
  /* load filter (1 outch x 1 height x 4 width) */        \
  float32x4_t vf00, vf01;                                 \
  vf00 = vld1q_f32(filter_ptr0);                          \
  vf01 = vld1q_f32(filter_ptr0 + 3);                      \
                                                          \
  /* outch 0 */                                           \
  vo0 = vmlaq_lane_f32(vo0, vi0, vget_low_f32(vf00), 0);  \
  vo0 = vmlaq_lane_f32(vo0, vi1, vget_low_f32(vf00), 1);  \
  vo0 = vmlaq_lane_f32(vo0, vi2, vget_high_f32(vf00), 0); \
  vo0 = vmlaq_lane_f32(vo0, vi3, vget_high_f32(vf00), 1); \
  vo0 = vmlaq_lane_f32(vo0, vi4, vget_low_f32(vf01), 1);  \
  vo0 = vmlaq_lane_f32(vo0, vi5, vget_high_f32(vf01), 0); \
  vo0 = vmlaq_lane_f32(vo0, vi6, vget_high_f32(vf01), 1);

MaceStatus Conv2dK7x7S1::Compute(const OpContext *context,
                                 const Tensor *input,
                                 const Tensor *filter,
                                 Tensor *output) {
  std::unique_ptr<const Tensor> padded_input;
  std::unique_ptr<Tensor> padded_output;
  ResizeOutAndPadInOut(context,
                       input,
                       filter,
                       output,
                       1,
                       4,
                       &padded_input,
                       &padded_output);
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
  auto filter_data = filter->data<float>();
  auto input_data = in_tensor->data<float>();
  auto output_data = out_tensor->mutable_data<float>();

  auto &in_shape = in_tensor->shape();
  auto &out_shape = out_tensor->shape();

  const index_t batch = in_shape[0];
  const index_t in_channels = in_shape[1];
  const index_t in_height = in_shape[2];
  const index_t in_width = in_shape[3];
  const index_t out_channels = out_shape[1];
  const index_t out_height = out_shape[2];
  const index_t out_width = out_shape[3];

  const index_t in_image_size = in_height * in_width;
  const index_t out_image_size = out_height * out_width;
  const index_t in_batch_size = in_channels * in_image_size;
  const index_t out_batch_size = out_channels * out_image_size;

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        if (m + 3 < out_channels) {
          float *out_ptr0_base =
              output_data + b * out_batch_size + m * out_image_size;
          float *out_ptr1_base =
              output_data + b * out_batch_size + (m + 1) * out_image_size;
          float *out_ptr2_base =
              output_data + b * out_batch_size + (m + 2) * out_image_size;
          float *out_ptr3_base =
              output_data + b * out_batch_size + (m + 3) * out_image_size;
          for (index_t c = 0; c < in_channels; ++c) {
            const float *in_ptr_base =
                input_data + b * in_batch_size + c * in_image_size;
            const float
                *filter_ptr0 = filter_data + m * in_channels * 49 + c * 49;
            const float *filter_ptr1 =
                filter_data + (m + 1) * in_channels * 49 + c * 49;
            const float *filter_ptr2 =
                filter_data + (m + 2) * in_channels * 49 + c * 49;
            const float *filter_ptr3 =
                filter_data + (m + 3) * in_channels * 49 + c * 49;
            for (index_t h = 0; h < out_height; ++h) {
              for (index_t w = 0; w + 3 < out_width; w += 4) {
                // input offset
                index_t in_offset = h * in_width + w;
                // output (4 outch x 1 height x 4 width): vo_outch_height
                float32x4_t vo0, vo1, vo2, vo3;
                // load output
                index_t out_offset = h * out_width + w;
                vo0 = vld1q_f32(out_ptr0_base + out_offset);
                vo1 = vld1q_f32(out_ptr1_base + out_offset);
                vo2 = vld1q_f32(out_ptr2_base + out_offset);
                vo3 = vld1q_f32(out_ptr3_base + out_offset);
                for (index_t r = 0; r < 7; ++r) {
                  // input (3 slide)
                  float32x4_t vi0, vi1, vi2, vi3, vi4, vi5, vi6;
                  float32x4_t vi8;  // for tmp use
                  // load input
                  vi0 = vld1q_f32(in_ptr_base + in_offset);
                  vi4 = vld1q_f32(in_ptr_base + in_offset + 4);
                  vi8 = vld1q_f32(in_ptr_base + in_offset + 8);
                  vi1 = vextq_f32(vi0, vi4, 1);
                  vi2 = vextq_f32(vi0, vi4, 2);
                  vi3 = vextq_f32(vi0, vi4, 3);
                  vi5 = vextq_f32(vi4, vi8, 1);
                  vi6 = vextq_f32(vi4, vi8, 2);

#if defined(__aarch64__)
                  MACE_Conv2dArmv8NeonK7x7SnLoadCalc4;
#else
                  MACE_Conv2dArmv7NeonK7x7SnLoadCalc4;
#endif

                  in_offset += in_width;
                  filter_ptr0 += 7;
                  filter_ptr1 += 7;
                  filter_ptr2 += 7;
                  filter_ptr3 += 7;
                }  // r

                vst1q_f32(out_ptr0_base + out_offset, vo0);
                vst1q_f32(out_ptr1_base + out_offset, vo1);
                vst1q_f32(out_ptr2_base + out_offset, vo2);
                vst1q_f32(out_ptr3_base + out_offset, vo3);

                filter_ptr0 -= 49;
                filter_ptr1 -= 49;
                filter_ptr2 -= 49;
                filter_ptr3 -= 49;
              }  // w
            }    // h
          }  // c
        } else {
          for (index_t mm = m; mm < out_channels; ++mm) {
            float *out_ptr0_base =
                output_data + b * out_batch_size + mm * out_image_size;
            for (index_t c = 0; c < in_channels; ++c) {
              const float *in_ptr_base =
                  input_data + b * in_batch_size + c * in_image_size;
              const float
                  *filter_ptr0 = filter_data + mm * in_channels * 49 + c * 49;
              for (index_t h = 0; h < out_height; ++h) {
                for (index_t w = 0; w + 3 < out_width; w += 4) {
                  // input offset
                  index_t in_offset = h * in_width + w;
                  // output (1 outch x 1 height x 4 width): vo_outch_height
                  float32x4_t vo0;
                  // load output
                  index_t out_offset = h * out_width + w;
                  vo0 = vld1q_f32(out_ptr0_base + out_offset);
                  for (index_t r = 0; r < 7; ++r) {
                    // input (3 slide)
                    float32x4_t vi0, vi1, vi2, vi3, vi4, vi5, vi6;
                    float32x4_t vi8;  // for tmp use
                    // load input
                    vi0 = vld1q_f32(in_ptr_base + in_offset);
                    vi4 = vld1q_f32(in_ptr_base + in_offset + 4);
                    vi8 = vld1q_f32(in_ptr_base + in_offset + 8);
                    vi1 = vextq_f32(vi0, vi4, 1);
                    vi2 = vextq_f32(vi0, vi4, 2);
                    vi3 = vextq_f32(vi0, vi4, 3);
                    vi5 = vextq_f32(vi4, vi8, 1);
                    vi6 = vextq_f32(vi4, vi8, 2);

#if defined(__aarch64__)
                    MACE_Conv2dArmv8NeonK7x7SnLoadCalc1;
#else
                    MACE_Conv2dArmv7NeonK7x7SnLoadCalc1;
#endif

                    in_offset += in_width;
                    filter_ptr0 += 7;
                  }  // r

                  vst1q_f32(out_ptr0_base + out_offset, vo0);
                  filter_ptr0 -= 49;
                }  // w
              }    // h
            }  // c
          }    // mm
        }      // if
      }        // m
    }          // b
  }, 0, batch, 1, 0, out_channels, 4);

  UnPadOutput(*out_tensor, output);
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Conv2dK7x7S2::Compute(const OpContext *context,
                                 const Tensor *input,
                                 const Tensor *filter,
                                 Tensor *output) {
  std::unique_ptr<const Tensor> padded_input;
  std::unique_ptr<Tensor> padded_output;
  ResizeOutAndPadInOut(context,
                       input,
                       filter,
                       output,
                       1,
                       4,
                       &padded_input,
                       &padded_output);
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
  auto filter_data = filter->data<float>();
  auto input_data = in_tensor->data<float>();
  auto output_data = out_tensor->mutable_data<float>();

  auto &in_shape = in_tensor->shape();
  auto &out_shape = out_tensor->shape();

  const index_t batch = in_shape[0];
  const index_t in_channels = in_shape[1];
  const index_t in_height = in_shape[2];
  const index_t in_width = in_shape[3];
  const index_t out_channels = out_shape[1];
  const index_t out_height = out_shape[2];
  const index_t out_width = out_shape[3];

  const index_t in_image_size = in_height * in_width;
  const index_t out_image_size = out_height * out_width;
  const index_t in_batch_size = in_channels * in_image_size;
  const index_t out_batch_size = out_channels * out_image_size;

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        if (m + 3 < out_channels) {
          float *out_ptr0_base =
              output_data + b * out_batch_size + m * out_image_size;
          float *out_ptr1_base =
              output_data + b * out_batch_size + (m + 1) * out_image_size;
          float *out_ptr2_base =
              output_data + b * out_batch_size + (m + 2) * out_image_size;
          float *out_ptr3_base =
              output_data + b * out_batch_size + (m + 3) * out_image_size;
          for (index_t c = 0; c < in_channels; ++c) {
            const float *in_ptr_base =
                input_data + b * in_batch_size + c * in_image_size;
            const float
                *filter_ptr0 = filter_data + m * in_channels * 49 + c * 49;
            const float *filter_ptr1 =
                filter_data + (m + 1) * in_channels * 49 + c * 49;
            const float *filter_ptr2 =
                filter_data + (m + 2) * in_channels * 49 + c * 49;
            const float *filter_ptr3 =
                filter_data + (m + 3) * in_channels * 49 + c * 49;
            for (index_t h = 0; h < out_height; ++h) {
              for (index_t w = 0; w + 3 < out_width; w += 4) {
                // input offset
                index_t in_h = h * 2;
                index_t in_w = w * 2;
                index_t in_offset = in_h * in_width + in_w;
                // output (4 outch x 1 height x 4 width): vo_outch_height
                float32x4_t vo0, vo1, vo2, vo3;
                // load output
                index_t out_offset = h * out_width + w;
                vo0 = vld1q_f32(out_ptr0_base + out_offset);
                vo1 = vld1q_f32(out_ptr1_base + out_offset);
                vo2 = vld1q_f32(out_ptr2_base + out_offset);
                vo3 = vld1q_f32(out_ptr3_base + out_offset);
                for (index_t r = 0; r < 7; ++r) {
                  // input (3 slide)
                  float32x4x2_t vvi0, vvi1;  // to de-interleave
                  float32x4_t vi0, vi1, vi2, vi3, vi4, vi5, vi6;
                  // load input
                  // [0.2.4.6, 1.3.5.7]
                  vvi0 = vld2q_f32(in_ptr_base + in_offset);
                  // [8.10.12.14, 9.11.13.15]
                  vvi1 = vld2q_f32(in_ptr_base + in_offset + 8);
                  vi0 = vvi0.val[0];                     // [0.2.4.6]
                  vi1 = vvi0.val[1];                     // [1.3.5.7]
                  vi2 = vextq_f32(vi0, vvi1.val[0], 1);  // [2.4.6.8]
                  vi3 = vextq_f32(vi1, vvi1.val[1], 1);  // [3.5.7.9]
                  vi4 = vextq_f32(vi0, vvi1.val[0], 2);  // [4.6.8.10]
                  vi5 = vextq_f32(vi1, vvi1.val[1], 2);  // [5.7.9.11]
                  vi6 = vextq_f32(vi0, vvi1.val[0], 3);  // [6.8.10.12]

#if defined(__aarch64__)
                  MACE_Conv2dArmv8NeonK7x7SnLoadCalc4;
#else
                  MACE_Conv2dArmv7NeonK7x7SnLoadCalc4;
#endif

                  in_offset += in_width;
                  filter_ptr0 += 7;
                  filter_ptr1 += 7;
                  filter_ptr2 += 7;
                  filter_ptr3 += 7;
                }  // r

                vst1q_f32(out_ptr0_base + out_offset, vo0);
                vst1q_f32(out_ptr1_base + out_offset, vo1);
                vst1q_f32(out_ptr2_base + out_offset, vo2);
                vst1q_f32(out_ptr3_base + out_offset, vo3);

                filter_ptr0 -= 49;
                filter_ptr1 -= 49;
                filter_ptr2 -= 49;
                filter_ptr3 -= 49;
              }  // w
            }    // h
          }  // c
        } else {
          for (index_t mm = m; mm < out_channels; ++mm) {
            float *out_ptr0_base =
                output_data + b * out_batch_size + mm * out_image_size;
            for (index_t c = 0; c < in_channels; ++c) {
              const float *in_ptr_base =
                  input_data + b * in_batch_size + c * in_image_size;
              const float
                  *filter_ptr0 = filter_data + mm * in_channels * 49 + c * 49;
              for (index_t h = 0; h < out_height; ++h) {
                for (index_t w = 0; w + 3 < out_width; w += 4) {
                  // input offset
                  index_t in_h = h * 2;
                  index_t in_w = w * 2;
                  index_t in_offset = in_h * in_width + in_w;
                  // output (1 outch x 1 height x 4 width): vo_outch_height
                  float32x4_t vo0;
                  // load ouput
                  index_t out_offset = h * out_width + w;
                  vo0 = vld1q_f32(out_ptr0_base + out_offset);
                  for (index_t r = 0; r < 7; ++r) {
                    // input (3 slide)
                    float32x4x2_t vvi0, vvi1;  // to de-interleave
                    float32x4_t vi0, vi1, vi2, vi3, vi4, vi5, vi6;
                    // load input
                    // [0.2.4.6, 1.3.5.7]
                    vvi0 = vld2q_f32(in_ptr_base + in_offset);
                    // [8.10.12.14, 9.11.13.15]
                    vvi1 = vld2q_f32(in_ptr_base + in_offset + 8);
                    vi0 = vvi0.val[0];                     // [0.2.4.6]
                    vi1 = vvi0.val[1];                     // [1.3.5.7]
                    vi2 = vextq_f32(vi0, vvi1.val[0], 1);  // [2.4.6.8]
                    vi3 = vextq_f32(vi1, vvi1.val[1], 1);  // [3.5.7.9]
                    vi4 = vextq_f32(vi0, vvi1.val[0], 2);  // [4.6.8.10]
                    vi5 = vextq_f32(vi1, vvi1.val[1], 2);  // [5.7.9.11]
                    vi6 = vextq_f32(vi0, vvi1.val[0], 3);  // [6.8.10.12]

#if defined(__aarch64__)
                    MACE_Conv2dArmv8NeonK7x7SnLoadCalc1;
#else
                    MACE_Conv2dArmv7NeonK7x7SnLoadCalc1;
#endif

                    in_offset += in_width;
                    filter_ptr0 += 7;
                  }  // r

                  vst1q_f32(out_ptr0_base + out_offset, vo0);
                  filter_ptr0 -= 49;
                }  // w
              }    // h
            }  // c
          }    // mm
        }      // if
      }        // m
    }          // b
  }, 0, batch, 1, 0, out_channels, 4);

  UnPadOutput(*out_tensor, output);
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Conv2dK7x7S3::Compute(const OpContext *context,
                                 const Tensor *input,
                                 const Tensor *filter,
                                 Tensor *output) {
  std::unique_ptr<const Tensor> padded_input;
  std::unique_ptr<Tensor> padded_output;
  ResizeOutAndPadInOut(context,
                       input,
                       filter,
                       output,
                       1,
                       4,
                       &padded_input,
                       &padded_output);
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
  auto filter_data = filter->data<float>();
  auto input_data = in_tensor->data<float>();
  auto output_data = out_tensor->mutable_data<float>();

  auto &in_shape = in_tensor->shape();
  auto &out_shape = out_tensor->shape();

  const index_t batch = in_shape[0];
  const index_t in_channels = in_shape[1];
  const index_t in_height = in_shape[2];
  const index_t in_width = in_shape[3];
  const index_t out_channels = out_shape[1];
  const index_t out_height = out_shape[2];
  const index_t out_width = out_shape[3];

  const index_t in_image_size = in_height * in_width;
  const index_t out_image_size = out_height * out_width;
  const index_t in_batch_size = in_channels * in_image_size;
  const index_t out_batch_size = out_channels * out_image_size;

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        if (m + 3 < out_channels) {
          float *out_ptr0_base =
              output_data + b * out_batch_size + m * out_image_size;
          float *out_ptr1_base =
              output_data + b * out_batch_size + (m + 1) * out_image_size;
          float *out_ptr2_base =
              output_data + b * out_batch_size + (m + 2) * out_image_size;
          float *out_ptr3_base =
              output_data + b * out_batch_size + (m + 3) * out_image_size;
          for (index_t c = 0; c < in_channels; ++c) {
            const float *in_ptr_base =
                input_data + b * in_batch_size + c * in_image_size;
            const float
                *filter_ptr0 = filter_data + m * in_channels * 49 + c * 49;
            const float *filter_ptr1 =
                filter_data + (m + 1) * in_channels * 49 + c * 49;
            const float *filter_ptr2 =
                filter_data + (m + 2) * in_channels * 49 + c * 49;
            const float *filter_ptr3 =
                filter_data + (m + 3) * in_channels * 49 + c * 49;
            for (index_t h = 0; h < out_height; ++h) {
              for (index_t w = 0; w + 3 < out_width; w += 4) {
                // input offset
                index_t in_h = h * 3;
                index_t in_w = w * 3;
                index_t in_offset = in_h * in_width + in_w;
                // output (4 outch x 1 height x 4 width): vo_outch_height
                float32x4_t vo0, vo1, vo2, vo3;
                // load output
                index_t out_offset = h * out_width + w;
                vo0 = vld1q_f32(out_ptr0_base + out_offset);
                vo1 = vld1q_f32(out_ptr1_base + out_offset);
                vo2 = vld1q_f32(out_ptr2_base + out_offset);
                vo3 = vld1q_f32(out_ptr3_base + out_offset);
                for (index_t r = 0; r < 7; ++r) {
                  // input (3 slide)
                  float32x4x3_t vvi0, vvi1;  // to de-interleave
                  float32x4_t vi0, vi1, vi2, vi3, vi4, vi5, vi6;
                  // load input
                  // [0.3.6.9, 1.4.7.10, 2.5.8.11]
                  vvi0 = vld3q_f32(in_ptr_base + in_offset);
                  // [12.15.xx.xx, 13.xx.xx.xx, 14.xx.xx.xx]
                  vvi1 = vld3q_f32(in_ptr_base + in_offset + 12);
                  vi0 = vvi0.val[0];                     // [0.3.6.9]
                  vi1 = vvi0.val[1];                     // [1.4.7.10]
                  vi2 = vvi0.val[2];                     // [2.5.8.11]
                  vi3 = vextq_f32(vi0, vvi1.val[0], 1);  // [3.6.9.12]
                  vi4 = vextq_f32(vi1, vvi1.val[1], 1);  // [4.7.10.13]
                  vi5 = vextq_f32(vi2, vvi1.val[2], 1);  // [5.8.11.14]
                  vi6 = vextq_f32(vi0, vvi1.val[0], 2);  // [6.9.12.15]

#if defined(__aarch64__)
                  MACE_Conv2dArmv8NeonK7x7SnLoadCalc4;
#else
                  MACE_Conv2dArmv7NeonK7x7SnLoadCalc4;
#endif

                  in_offset += in_width;
                  filter_ptr0 += 7;
                  filter_ptr1 += 7;
                  filter_ptr2 += 7;
                  filter_ptr3 += 7;
                }  // r

                vst1q_f32(out_ptr0_base + out_offset, vo0);
                vst1q_f32(out_ptr1_base + out_offset, vo1);
                vst1q_f32(out_ptr2_base + out_offset, vo2);
                vst1q_f32(out_ptr3_base + out_offset, vo3);

                filter_ptr0 -= 49;
                filter_ptr1 -= 49;
                filter_ptr2 -= 49;
                filter_ptr3 -= 49;
              }  // w
            }    // h
          }  // c
        } else {
          for (index_t mm = m; mm < out_channels; ++mm) {
            float *out_ptr0_base =
                output_data + b * out_batch_size + mm * out_image_size;
            for (index_t c = 0; c < in_channels; ++c) {
              const float *in_ptr_base =
                  input_data + b * in_batch_size + c * in_image_size;
              const float
                  *filter_ptr0 = filter_data + mm * in_channels * 49 + c * 49;
              for (index_t h = 0; h < out_height; ++h) {
                for (index_t w = 0; w + 3 < out_width; w += 4) {
                  // input offset
                  index_t in_h = h * 3;
                  index_t in_w = w * 3;
                  index_t in_offset = in_h * in_width + in_w;
                  // output (1 outch x 1 height x 4 width): vo_outch_height
                  float32x4_t vo0;
                  // load output
                  index_t out_offset = h * out_width + w;
                  vo0 = vld1q_f32(out_ptr0_base + out_offset);
                  for (index_t r = 0; r < 7; ++r) {
                    // input (3 slide)
                    float32x4x3_t vvi0, vvi1;  // to de-interleave
                    float32x4_t vi0, vi1, vi2, vi3, vi4, vi5, vi6;
                    // load input
                    // [0.3.6.9, 1.4.7.10, 2.5.8.11]
                    vvi0 = vld3q_f32(in_ptr_base + in_offset);
                    // [12.15.xx.xx, 13.xx.xx.xx, 14.xx.xx.xx]
                    vvi1 = vld3q_f32(in_ptr_base + in_offset + 12);
                    vi0 = vvi0.val[0];                     // [0.3.6.9]
                    vi1 = vvi0.val[1];                     // [1.4.7.10]
                    vi2 = vvi0.val[2];                     // [2.5.8.11]
                    vi3 = vextq_f32(vi0, vvi1.val[0], 1);  // [3.6.9.12]
                    vi4 = vextq_f32(vi1, vvi1.val[1], 1);  // [4.7.10.13]
                    vi5 = vextq_f32(vi2, vvi1.val[2], 1);  // [5.8.11.14]
                    vi6 = vextq_f32(vi0, vvi1.val[0], 2);  // [6.9.12.15]

#if defined(__aarch64__)
                    MACE_Conv2dArmv8NeonK7x7SnLoadCalc1;
#else
                    MACE_Conv2dArmv7NeonK7x7SnLoadCalc1;
#endif

                    in_offset += in_width;
                    filter_ptr0 += 7;
                  }  // r

                  vst1q_f32(out_ptr0_base + out_offset, vo0);
                  filter_ptr0 -= 49;
                }  // w
              }    // h
            }  // c
          }    // mm
        }      // if
      }        // m
    }          // b
  }, 0, batch, 1, 0, out_channels, 4);

  UnPadOutput(*out_tensor, output);
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace
