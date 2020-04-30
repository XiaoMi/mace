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

#include <arm_neon.h>
#include <memory>

#include "mace/ops/arm/base/conv_2d_5x5.h"
#include "mace/ops/delegator/conv_2d.h"

namespace mace {
namespace ops {
namespace arm {

#define MACE_Conv2dNeonK5x5SnLoadCalc4                    \
  /* load filter (4 outch x 1 height x 4 width) */        \
  float32x4_t vf00, vf10, vf20, vf30;                     \
  float32x2_t vf01, vf11, vf21, vf31;                     \
  vf00 = vld1q_f32(filter_ptr0);                          \
  vf01 = vld1_f32(filter_ptr0 + 3);                       \
  vf10 = vld1q_f32(filter_ptr1);                          \
  vf11 = vld1_f32(filter_ptr1 + 3);                       \
  vf20 = vld1q_f32(filter_ptr2);                          \
  vf21 = vld1_f32(filter_ptr2 + 3);                       \
  vf30 = vld1q_f32(filter_ptr3);                          \
  vf31 = vld1_f32(filter_ptr3 + 3);                       \
                                                          \
  /* outch 0 */                                           \
  vo0 = vmlaq_lane_f32(vo0, vi0, vget_low_f32(vf00), 0);  \
  vo0 = vmlaq_lane_f32(vo0, vi1, vget_low_f32(vf00), 1);  \
  vo0 = vmlaq_lane_f32(vo0, vi2, vget_high_f32(vf00), 0); \
  vo0 = vmlaq_lane_f32(vo0, vi3, vget_high_f32(vf00), 1); \
  vo0 = vmlaq_lane_f32(vo0, vi4, vf01, 1);                \
                                                          \
  /* outch 1 */                                           \
  vo1 = vmlaq_lane_f32(vo1, vi0, vget_low_f32(vf10), 0);  \
  vo1 = vmlaq_lane_f32(vo1, vi1, vget_low_f32(vf10), 1);  \
  vo1 = vmlaq_lane_f32(vo1, vi2, vget_high_f32(vf10), 0); \
  vo1 = vmlaq_lane_f32(vo1, vi3, vget_high_f32(vf10), 1); \
  vo1 = vmlaq_lane_f32(vo1, vi4, vf11, 1);                \
                                                          \
  /* outch 2 */                                           \
  vo2 = vmlaq_lane_f32(vo2, vi0, vget_low_f32(vf20), 0);  \
  vo2 = vmlaq_lane_f32(vo2, vi1, vget_low_f32(vf20), 1);  \
  vo2 = vmlaq_lane_f32(vo2, vi2, vget_high_f32(vf20), 0); \
  vo2 = vmlaq_lane_f32(vo2, vi3, vget_high_f32(vf20), 1); \
  vo2 = vmlaq_lane_f32(vo2, vi4, vf21, 1);                \
                                                          \
  /* outch 3 */                                           \
  vo3 = vmlaq_lane_f32(vo3, vi0, vget_low_f32(vf30), 0);  \
  vo3 = vmlaq_lane_f32(vo3, vi1, vget_low_f32(vf30), 1);  \
  vo3 = vmlaq_lane_f32(vo3, vi2, vget_high_f32(vf30), 0); \
  vo3 = vmlaq_lane_f32(vo3, vi3, vget_high_f32(vf30), 1); \
  vo3 = vmlaq_lane_f32(vo3, vi4, vf31, 1);

#define MACE_Conv2dNeonK5x5SnLoadCalc1                    \
  /* load filter (1 outch x 1 height x 4 width) */        \
  float32x4_t vf00;                                       \
  float32x2_t vf01;                                       \
  vf00 = vld1q_f32(filter_ptr0);                          \
  vf01 = vld1_f32(filter_ptr0 + 3);                       \
                                                          \
  /* outch 0 */                                           \
  vo0 = vmlaq_lane_f32(vo0, vi0, vget_low_f32(vf00), 0);  \
  vo0 = vmlaq_lane_f32(vo0, vi1, vget_low_f32(vf00), 1);  \
  vo0 = vmlaq_lane_f32(vo0, vi2, vget_high_f32(vf00), 0); \
  vo0 = vmlaq_lane_f32(vo0, vi3, vget_high_f32(vf00), 1); \
  vo0 = vmlaq_lane_f32(vo0, vi4, vf01, 1);

template<>
MaceStatus Conv2dK5x5S1<float>::DoCompute(
    const ConvComputeParam &p, const float *filter_data,
    const float *input_data, float *output_data) {
  p.thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        if (m + 3 < p.out_channels) {
          float *out_ptr0_base =
              output_data + b * p.out_batch_size + m * p.out_image_size;
          float *out_ptr1_base =
              output_data + b * p.out_batch_size + (m + 1) * p.out_image_size;
          float *out_ptr2_base =
              output_data + b * p.out_batch_size + (m + 2) * p.out_image_size;
          float *out_ptr3_base =
              output_data + b * p.out_batch_size + (m + 3) * p.out_image_size;

          for (index_t c = 0; c < p.in_channels; ++c) {
            const float *in_ptr_base =
                input_data + b * p.in_batch_size + c * p.in_image_size;
            const float
                *filter_ptr0 = filter_data + m * p.in_channels * 25 + c * 25;
            const float *filter_ptr1 =
                filter_data + (m + 1) * p.in_channels * 25 + c * 25;
            const float *filter_ptr2 =
                filter_data + (m + 2) * p.in_channels * 25 + c * 25;
            const float *filter_ptr3 =
                filter_data + (m + 3) * p.in_channels * 25 + c * 25;
            for (index_t h = 0; h < p.out_height; ++h) {
              for (index_t w = 0; w + 3 < p.out_width; w += 4) {
                // input offset
                index_t in_offset = h * p.in_width + w;
                // output (4 outch x 1 height x 4 width): vo_outch_height
                float32x4_t vo0, vo1, vo2, vo3;
                // load output
                index_t out_offset = h * p.out_width + w;
                vo0 = vld1q_f32(out_ptr0_base + out_offset);
                vo1 = vld1q_f32(out_ptr1_base + out_offset);
                vo2 = vld1q_f32(out_ptr2_base + out_offset);
                vo3 = vld1q_f32(out_ptr3_base + out_offset);
                for (index_t r = 0; r < 5; ++r) {
                  // input (3 slide)
                  float32x4_t vi0, vi1, vi2, vi3, vi4;
                  // load input
                  vi0 = vld1q_f32(in_ptr_base + in_offset);
                  vi4 = vld1q_f32(in_ptr_base + in_offset + 4);
                  vi1 = vextq_f32(vi0, vi4, 1);
                  vi2 = vextq_f32(vi0, vi4, 2);
                  vi3 = vextq_f32(vi0, vi4, 3);

                  MACE_Conv2dNeonK5x5SnLoadCalc4;

                  in_offset += p.in_width;
                  filter_ptr0 += 5;
                  filter_ptr1 += 5;
                  filter_ptr2 += 5;
                  filter_ptr3 += 5;
                }  // r

                vst1q_f32(out_ptr0_base + out_offset, vo0);
                vst1q_f32(out_ptr1_base + out_offset, vo1);
                vst1q_f32(out_ptr2_base + out_offset, vo2);
                vst1q_f32(out_ptr3_base + out_offset, vo3);

                filter_ptr0 -= 25;
                filter_ptr1 -= 25;
                filter_ptr2 -= 25;
                filter_ptr3 -= 25;
              }  // w
            }    // h
          }  // c
        } else {
          for (index_t mm = m; mm < p.out_channels; ++mm) {
            float *out_ptr0_base =
                output_data + b * p.out_batch_size + mm * p.out_image_size;
            for (index_t c = 0; c < p.in_channels; ++c) {
              const float *in_ptr_base =
                  input_data + b * p.in_batch_size + c * p.in_image_size;
              const float
                  *filter_ptr0 = filter_data + mm * p.in_channels * 25 + c * 25;
              for (index_t h = 0; h < p.out_height; ++h) {
                for (index_t w = 0; w + 3 < p.out_width; w += 4) {
                  // input offset
                  index_t in_offset = h * p.in_width + w;
                  // output (1 outch x 1 height x 4 width): vo_outch_height
                  float32x4_t vo0;
                  // load output
                  index_t out_offset = h * p.out_width + w;
                  vo0 = vld1q_f32(out_ptr0_base + out_offset);
                  for (index_t r = 0; r < 5; ++r) {
                    // input (3 slide)
                    float32x4_t vi0, vi1, vi2, vi3, vi4;
                    // load input
                    vi0 = vld1q_f32(in_ptr_base + in_offset);
                    vi4 = vld1q_f32(in_ptr_base + in_offset + 4);
                    vi1 = vextq_f32(vi0, vi4, 1);
                    vi2 = vextq_f32(vi0, vi4, 2);
                    vi3 = vextq_f32(vi0, vi4, 3);

                    MACE_Conv2dNeonK5x5SnLoadCalc1;

                    in_offset += p.in_width;
                    filter_ptr0 += 5;
                  }  // r

                  vst1q_f32(out_ptr0_base + out_offset, vo0);
                  filter_ptr0 -= 25;
                }  // w
              }    // h
            }  // c
          }    // mm
        }      // if
      }        // m
    }          // b
  }, 0, p.batch, 1, 0, p.out_channels, 4);

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
