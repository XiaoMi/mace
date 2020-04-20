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

#include <arm_neon.h>
#include <memory>

#include "mace/ops/arm/base/conv_2d_1xn.h"
#include "mace/ops/delegator/conv_2d.h"

namespace mace {
namespace ops {
namespace arm {

template<>
MaceStatus Conv2dK1x7S1<float>::DoCompute(
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
            const float *filter_ptr0 =
                filter_data + m * p.in_channels * 7 + c * 7;
            const float *filter_ptr1 =
                filter_data + (m + 1) * p.in_channels * 7 + c * 7;
            const float *filter_ptr2 =
                filter_data + (m + 2) * p.in_channels * 7 + c * 7;
            const float *filter_ptr3 =
                filter_data + (m + 3) * p.in_channels * 7 + c * 7;
            /* load filter (4 outch x 1 height x 4 width) */
            float32x4_t vf00, vf01;
            float32x4_t vf10, vf11;
            float32x4_t vf20, vf21;
            float32x4_t vf30, vf31;
            vf00 = vld1q_f32(filter_ptr0);
            vf01 = vld1q_f32(filter_ptr0 + 3);
            vf10 = vld1q_f32(filter_ptr1);
            vf11 = vld1q_f32(filter_ptr1 + 3);
            vf20 = vld1q_f32(filter_ptr2);
            vf21 = vld1q_f32(filter_ptr2 + 3);
            vf30 = vld1q_f32(filter_ptr3);
            vf31 = vld1q_f32(filter_ptr3 + 3);

            for (index_t h = 0; h < p.out_height; ++h) {
              for (index_t w = 0; w + 3 < p.out_width; w += 4) {
                // output (4 outch x 1 height x 4 width): vo_outch_height
                float32x4_t vo0, vo1, vo2, vo3;
                // load output
                index_t out_offset = h * p.out_width + w;
                vo0 = vld1q_f32(out_ptr0_base + out_offset);
                vo1 = vld1q_f32(out_ptr1_base + out_offset);
                vo2 = vld1q_f32(out_ptr2_base + out_offset);
                vo3 = vld1q_f32(out_ptr3_base + out_offset);

                // input (3 slide)
                float32x4_t vi0, vi1, vi2, vi3, vi4, vi5, vi6, vi8;
                // input offset
                index_t in_offset = h * p.in_width + w;
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
                /* outch 0 */
              vo0 = vfmaq_laneq_f32(vo0, vi0, vf00, 0);
              vo0 = vfmaq_laneq_f32(vo0, vi1, vf00, 1);
              vo0 = vfmaq_laneq_f32(vo0, vi2, vf00, 2);
              vo0 = vfmaq_laneq_f32(vo0, vi3, vf00, 3);
              vo0 = vfmaq_laneq_f32(vo0, vi4, vf01, 1);
              vo0 = vfmaq_laneq_f32(vo0, vi5, vf01, 2);
              vo0 = vfmaq_laneq_f32(vo0, vi6, vf01, 3);
              /* outch 1 */
              vo1 = vfmaq_laneq_f32(vo1, vi0, vf10, 0);
              vo1 = vfmaq_laneq_f32(vo1, vi1, vf10, 1);
              vo1 = vfmaq_laneq_f32(vo1, vi2, vf10, 2);
              vo1 = vfmaq_laneq_f32(vo1, vi3, vf10, 3);
              vo1 = vfmaq_laneq_f32(vo1, vi4, vf11, 1);
              vo1 = vfmaq_laneq_f32(vo1, vi5, vf11, 2);
              vo1 = vfmaq_laneq_f32(vo1, vi6, vf11, 3);
              /* outch 2 */
              vo2 = vfmaq_laneq_f32(vo2, vi0, vf20, 0);
              vo2 = vfmaq_laneq_f32(vo2, vi1, vf20, 1);
              vo2 = vfmaq_laneq_f32(vo2, vi2, vf20, 2);
              vo2 = vfmaq_laneq_f32(vo2, vi3, vf20, 3);
              vo2 = vfmaq_laneq_f32(vo2, vi4, vf21, 1);
              vo2 = vfmaq_laneq_f32(vo2, vi5, vf21, 2);
              vo2 = vfmaq_laneq_f32(vo2, vi6, vf21, 3);
              /* outch 3 */
              vo3 = vfmaq_laneq_f32(vo3, vi0, vf30, 0);
              vo3 = vfmaq_laneq_f32(vo3, vi1, vf30, 1);
              vo3 = vfmaq_laneq_f32(vo3, vi2, vf30, 2);
              vo3 = vfmaq_laneq_f32(vo3, vi3, vf30, 3);
              vo3 = vfmaq_laneq_f32(vo3, vi4, vf31, 1);
              vo3 = vfmaq_laneq_f32(vo3, vi5, vf31, 2);
              vo3 = vfmaq_laneq_f32(vo3, vi6, vf31, 3);
#else
                /* outch 0 */
                vo0 = vmlaq_lane_f32(vo0, vi0, vget_low_f32(vf00), 0);
                vo0 = vmlaq_lane_f32(vo0, vi1, vget_low_f32(vf00), 1);
                vo0 = vmlaq_lane_f32(vo0, vi2, vget_high_f32(vf00), 0);
                vo0 = vmlaq_lane_f32(vo0, vi3, vget_high_f32(vf00), 1);
                vo0 = vmlaq_lane_f32(vo0, vi4, vget_low_f32(vf01), 1);
                vo0 = vmlaq_lane_f32(vo0, vi5, vget_high_f32(vf01), 0);
                vo0 = vmlaq_lane_f32(vo0, vi6, vget_high_f32(vf01), 1);
                /* outch 1 */
                vo1 = vmlaq_lane_f32(vo1, vi0, vget_low_f32(vf10), 0);
                vo1 = vmlaq_lane_f32(vo1, vi1, vget_low_f32(vf10), 1);
                vo1 = vmlaq_lane_f32(vo1, vi2, vget_high_f32(vf10), 0);
                vo1 = vmlaq_lane_f32(vo1, vi3, vget_high_f32(vf10), 1);
                vo1 = vmlaq_lane_f32(vo1, vi4, vget_low_f32(vf11), 1);
                vo1 = vmlaq_lane_f32(vo1, vi5, vget_high_f32(vf11), 0);
                vo1 = vmlaq_lane_f32(vo1, vi6, vget_high_f32(vf11), 1);
                /* outch 2 */
                vo2 = vmlaq_lane_f32(vo2, vi0, vget_low_f32(vf20), 0);
                vo2 = vmlaq_lane_f32(vo2, vi1, vget_low_f32(vf20), 1);
                vo2 = vmlaq_lane_f32(vo2, vi2, vget_high_f32(vf20), 0);
                vo2 = vmlaq_lane_f32(vo2, vi3, vget_high_f32(vf20), 1);
                vo2 = vmlaq_lane_f32(vo2, vi4, vget_low_f32(vf21), 1);
                vo2 = vmlaq_lane_f32(vo2, vi5, vget_high_f32(vf21), 0);
                vo2 = vmlaq_lane_f32(vo2, vi6, vget_high_f32(vf21), 1);
                /* outch 3 */
                vo3 = vmlaq_lane_f32(vo3, vi0, vget_low_f32(vf30), 0);
                vo3 = vmlaq_lane_f32(vo3, vi1, vget_low_f32(vf30), 1);
                vo3 = vmlaq_lane_f32(vo3, vi2, vget_high_f32(vf30), 0);
                vo3 = vmlaq_lane_f32(vo3, vi3, vget_high_f32(vf30), 1);
                vo3 = vmlaq_lane_f32(vo3, vi4, vget_low_f32(vf31), 1);
                vo3 = vmlaq_lane_f32(vo3, vi5, vget_high_f32(vf31), 0);
                vo3 = vmlaq_lane_f32(vo3, vi6, vget_high_f32(vf31), 1);
#endif

                vst1q_f32(out_ptr0_base + out_offset, vo0);
                vst1q_f32(out_ptr1_base + out_offset, vo1);
                vst1q_f32(out_ptr2_base + out_offset, vo2);
                vst1q_f32(out_ptr3_base + out_offset, vo3);
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
              const float *filter_ptr0 =
                  filter_data + mm * p.in_channels * 7 + c * 7;
              /* load filter (1 outch x 1 height x 4 width) */
              float32x4_t vf00, vf01;
              vf00 = vld1q_f32(filter_ptr0);
              vf01 = vld1q_f32(filter_ptr0 + 3);

              for (index_t h = 0; h < p.out_height; ++h) {
                for (index_t w = 0; w + 3 < p.out_width; w += 4) {
                  // output (1 outch x 1 height x 4 width): vo_outch_height
                  float32x4_t vo0;
                  // load output
                  index_t out_offset = h * p.out_width + w;
                  vo0 = vld1q_f32(out_ptr0_base + out_offset);

                  // input (3 slide)
                  float32x4_t vi0, vi1, vi2, vi3, vi4, vi5, vi6, vi8;
                  // input offset
                  index_t in_offset = h * p.in_width + w;
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
                  vo0 = vfmaq_laneq_f32(vo0, vi0, vf00, 0);
                vo0 = vfmaq_laneq_f32(vo0, vi1, vf00, 1);
                vo0 = vfmaq_laneq_f32(vo0, vi2, vf00, 2);
                vo0 = vfmaq_laneq_f32(vo0, vi3, vf00, 3);
                vo0 = vfmaq_laneq_f32(vo0, vi4, vf01, 1);
                vo0 = vfmaq_laneq_f32(vo0, vi5, vf01, 2);
                vo0 = vfmaq_laneq_f32(vo0, vi6, vf01, 3);
#else
                  vo0 = vmlaq_lane_f32(vo0, vi0, vget_low_f32(vf00), 0);
                  vo0 = vmlaq_lane_f32(vo0, vi1, vget_low_f32(vf00), 1);
                  vo0 = vmlaq_lane_f32(vo0, vi2, vget_high_f32(vf00), 0);
                  vo0 = vmlaq_lane_f32(vo0, vi3, vget_high_f32(vf00), 1);
                  vo0 = vmlaq_lane_f32(vo0, vi4, vget_low_f32(vf01), 1);
                  vo0 = vmlaq_lane_f32(vo0, vi5, vget_high_f32(vf01), 0);
                  vo0 = vmlaq_lane_f32(vo0, vi6, vget_high_f32(vf01), 1);
#endif

                  vst1q_f32(out_ptr0_base + out_offset, vo0);
                }  // w
              }    // h
            }  // c
          }
        }  // if
      }    // m
    }      // b
  }, 0, p.batch, 1, 0, p.out_channels, 4);

  return MaceStatus::MACE_SUCCESS;
}

template<>
MaceStatus Conv2dK7x1S1<float>::DoCompute(
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
            const float *filter_ptr0 =
                filter_data + m * p.in_channels * 7 + c * 7;
            const float *filter_ptr1 =
                filter_data + (m + 1) * p.in_channels * 7 + c * 7;
            const float *filter_ptr2 =
                filter_data + (m + 2) * p.in_channels * 7 + c * 7;
            const float *filter_ptr3 =
                filter_data + (m + 3) * p.in_channels * 7 + c * 7;
            /* load filter (4 outch x 4 height x 1 width) */
            float32x4_t vf00, vf01;
            float32x4_t vf10, vf11;
            float32x4_t vf20, vf21;
            float32x4_t vf30, vf31;
            vf00 = vld1q_f32(filter_ptr0);
            vf01 = vld1q_f32(filter_ptr0 + 3);
            vf10 = vld1q_f32(filter_ptr1);
            vf11 = vld1q_f32(filter_ptr1 + 3);
            vf20 = vld1q_f32(filter_ptr2);
            vf21 = vld1q_f32(filter_ptr2 + 3);
            vf30 = vld1q_f32(filter_ptr3);
            vf31 = vld1q_f32(filter_ptr3 + 3);

            for (index_t h = 0; h + 3 < p.out_height; h += 4) {
              for (index_t w = 0; w < p.out_width; ++w) {
                // load output
                index_t out_offset = h * p.out_width + w;
                // output (4 outch x 4 height x 1 width): vo_outch_height
                float32x4_t vo0 = {out_ptr0_base[out_offset],
                                   out_ptr0_base[out_offset + p.out_width],
                                   out_ptr0_base[out_offset + 2 * p.out_width],
                                   out_ptr0_base[out_offset + 3 * p.out_width]};
                float32x4_t vo1 = {out_ptr1_base[out_offset],
                                   out_ptr1_base[out_offset + p.out_width],
                                   out_ptr1_base[out_offset + 2 * p.out_width],
                                   out_ptr1_base[out_offset + 3 * p.out_width]};
                float32x4_t vo2 = {out_ptr2_base[out_offset],
                                   out_ptr2_base[out_offset + p.out_width],
                                   out_ptr2_base[out_offset + 2 * p.out_width],
                                   out_ptr2_base[out_offset + 3 * p.out_width]};
                float32x4_t vo3 = {out_ptr3_base[out_offset],
                                   out_ptr3_base[out_offset + p.out_width],
                                   out_ptr3_base[out_offset + 2 * p.out_width],
                                   out_ptr3_base[out_offset + 3 * p.out_width]};

                // input offset
                index_t in_offset = h * p.in_width + w;
                // input (3 slide)
                float32x4_t vi0 = {in_ptr_base[in_offset],
                                   in_ptr_base[in_offset + p.in_width],
                                   in_ptr_base[in_offset + 2 * p.in_width],
                                   in_ptr_base[in_offset + 3 * p.in_width]};
                float32x4_t vi4 = {in_ptr_base[in_offset + 4 * p.in_width],
                                   in_ptr_base[in_offset + 5 * p.in_width],
                                   in_ptr_base[in_offset + 6 * p.in_width],
                                   in_ptr_base[in_offset + 7 * p.in_width]};
                float32x4_t vi8 = {in_ptr_base[in_offset + 8 * p.in_width],
                                   in_ptr_base[in_offset + 9 * p.in_width]};
                float32x4_t vi1 = vextq_f32(vi0, vi4, 1);
                float32x4_t vi2 = vextq_f32(vi0, vi4, 2);
                float32x4_t vi3 = vextq_f32(vi0, vi4, 3);
                float32x4_t vi5 = vextq_f32(vi4, vi8, 1);
                float32x4_t vi6 = vextq_f32(vi4, vi8, 2);

#if defined(__aarch64__)
                /* outch 0 */
                vo0 = vfmaq_laneq_f32(vo0, vi0, vf00, 0);
                vo0 = vfmaq_laneq_f32(vo0, vi1, vf00, 1);
                vo0 = vfmaq_laneq_f32(vo0, vi2, vf00, 2);
                vo0 = vfmaq_laneq_f32(vo0, vi3, vf00, 3);
                vo0 = vfmaq_laneq_f32(vo0, vi4, vf01, 1);
                vo0 = vfmaq_laneq_f32(vo0, vi5, vf01, 2);
                vo0 = vfmaq_laneq_f32(vo0, vi6, vf01, 3);
                /* outch 1 */
                vo1 = vfmaq_laneq_f32(vo1, vi0, vf10, 0);
                vo1 = vfmaq_laneq_f32(vo1, vi1, vf10, 1);
                vo1 = vfmaq_laneq_f32(vo1, vi2, vf10, 2);
                vo1 = vfmaq_laneq_f32(vo1, vi3, vf10, 3);
                vo1 = vfmaq_laneq_f32(vo1, vi4, vf11, 1);
                vo1 = vfmaq_laneq_f32(vo1, vi5, vf11, 2);
                vo1 = vfmaq_laneq_f32(vo1, vi6, vf11, 3);
                /* outch 2 */
                vo2 = vfmaq_laneq_f32(vo2, vi0, vf20, 0);
                vo2 = vfmaq_laneq_f32(vo2, vi1, vf20, 1);
                vo2 = vfmaq_laneq_f32(vo2, vi2, vf20, 2);
                vo2 = vfmaq_laneq_f32(vo2, vi3, vf20, 3);
                vo2 = vfmaq_laneq_f32(vo2, vi4, vf21, 1);
                vo2 = vfmaq_laneq_f32(vo2, vi5, vf21, 2);
                vo2 = vfmaq_laneq_f32(vo2, vi6, vf21, 3);
                /* outch 3 */
                vo3 = vfmaq_laneq_f32(vo3, vi0, vf30, 0);
                vo3 = vfmaq_laneq_f32(vo3, vi1, vf30, 1);
                vo3 = vfmaq_laneq_f32(vo3, vi2, vf30, 2);
                vo3 = vfmaq_laneq_f32(vo3, vi3, vf30, 3);
                vo3 = vfmaq_laneq_f32(vo3, vi4, vf31, 1);
                vo3 = vfmaq_laneq_f32(vo3, vi5, vf31, 2);
                vo3 = vfmaq_laneq_f32(vo3, vi6, vf31, 3);
#else
                /* outch 0 */
                vo0 = vmlaq_lane_f32(vo0, vi0, vget_low_f32(vf00), 0);
                vo0 = vmlaq_lane_f32(vo0, vi1, vget_low_f32(vf00), 1);
                vo0 = vmlaq_lane_f32(vo0, vi2, vget_high_f32(vf00), 0);
                vo0 = vmlaq_lane_f32(vo0, vi3, vget_high_f32(vf00), 1);
                vo0 = vmlaq_lane_f32(vo0, vi4, vget_low_f32(vf01), 1);
                vo0 = vmlaq_lane_f32(vo0, vi5, vget_high_f32(vf01), 0);
                vo0 = vmlaq_lane_f32(vo0, vi6, vget_high_f32(vf01), 1);
                /* outch 1 */
                vo1 = vmlaq_lane_f32(vo1, vi0, vget_low_f32(vf10), 0);
                vo1 = vmlaq_lane_f32(vo1, vi1, vget_low_f32(vf10), 1);
                vo1 = vmlaq_lane_f32(vo1, vi2, vget_high_f32(vf10), 0);
                vo1 = vmlaq_lane_f32(vo1, vi3, vget_high_f32(vf10), 1);
                vo1 = vmlaq_lane_f32(vo1, vi4, vget_low_f32(vf11), 1);
                vo1 = vmlaq_lane_f32(vo1, vi5, vget_high_f32(vf11), 0);
                vo1 = vmlaq_lane_f32(vo1, vi6, vget_high_f32(vf11), 1);
                /* outch 2 */
                vo2 = vmlaq_lane_f32(vo2, vi0, vget_low_f32(vf20), 0);
                vo2 = vmlaq_lane_f32(vo2, vi1, vget_low_f32(vf20), 1);
                vo2 = vmlaq_lane_f32(vo2, vi2, vget_high_f32(vf20), 0);
                vo2 = vmlaq_lane_f32(vo2, vi3, vget_high_f32(vf20), 1);
                vo2 = vmlaq_lane_f32(vo2, vi4, vget_low_f32(vf21), 1);
                vo2 = vmlaq_lane_f32(vo2, vi5, vget_high_f32(vf21), 0);
                vo2 = vmlaq_lane_f32(vo2, vi6, vget_high_f32(vf21), 1);
                /* outch 3 */
                vo3 = vmlaq_lane_f32(vo3, vi0, vget_low_f32(vf30), 0);
                vo3 = vmlaq_lane_f32(vo3, vi1, vget_low_f32(vf30), 1);
                vo3 = vmlaq_lane_f32(vo3, vi2, vget_high_f32(vf30), 0);
                vo3 = vmlaq_lane_f32(vo3, vi3, vget_high_f32(vf30), 1);
                vo3 = vmlaq_lane_f32(vo3, vi4, vget_low_f32(vf31), 1);
                vo3 = vmlaq_lane_f32(vo3, vi5, vget_high_f32(vf31), 0);
                vo3 = vmlaq_lane_f32(vo3, vi6, vget_high_f32(vf31), 1);
#endif

                out_ptr0_base[out_offset] = vo0[0];
                out_ptr0_base[out_offset + p.out_width] = vo0[1];
                out_ptr0_base[out_offset + 2 * p.out_width] = vo0[2];
                out_ptr0_base[out_offset + 3 * p.out_width] = vo0[3];
                out_ptr1_base[out_offset] = vo1[0];
                out_ptr1_base[out_offset + p.out_width] = vo1[1];
                out_ptr1_base[out_offset + 2 * p.out_width] = vo1[2];
                out_ptr1_base[out_offset + 3 * p.out_width] = vo1[3];
                out_ptr2_base[out_offset] = vo2[0];
                out_ptr2_base[out_offset + p.out_width] = vo2[1];
                out_ptr2_base[out_offset + 2 * p.out_width] = vo2[2];
                out_ptr2_base[out_offset + 3 * p.out_width] = vo2[3];
                out_ptr3_base[out_offset] = vo3[0];
                out_ptr3_base[out_offset + p.out_width] = vo3[1];
                out_ptr3_base[out_offset + 2 * p.out_width] = vo3[2];
                out_ptr3_base[out_offset + 3 * p.out_width] = vo3[3];
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
              const float *filter_ptr0 =
                  filter_data + mm * p.in_channels * 7 + c * 7;
              /* load filter (1 outch x 4 height x 1 width) */
              float32x4_t vf00, vf01;
              vf00 = vld1q_f32(filter_ptr0);
              vf01 = vld1q_f32(filter_ptr0 + 3);

              for (index_t h = 0; h + 3 < p.out_height; h += 4) {
                for (index_t w = 0; w < p.out_width; ++w) {
                  // load output
                  index_t out_offset = h * p.out_width + w;
                  // output (1 outch x 4 height x 1 width): vo_outch_height
                  float32x4_t vo0 = {out_ptr0_base[out_offset],
                                     out_ptr0_base[out_offset + p.out_width],
                                     out_ptr0_base[out_offset
                                         + 2 * p.out_width],
                                     out_ptr0_base[out_offset
                                         + 3 * p.out_width]};

                  // input offset
                  index_t in_offset = h * p.in_width + w;
                  // input (3 slide)
                  float32x4_t vi0 = {in_ptr_base[in_offset],
                                     in_ptr_base[in_offset + p.in_width],
                                     in_ptr_base[in_offset + 2 * p.in_width],
                                     in_ptr_base[in_offset + 3 * p.in_width]};
                  float32x4_t vi4 = {in_ptr_base[in_offset + 4 * p.in_width],
                                     in_ptr_base[in_offset + 5 * p.in_width],
                                     in_ptr_base[in_offset + 6 * p.in_width],
                                     in_ptr_base[in_offset + 7 * p.in_width]};
                  float32x4_t vi8 = {in_ptr_base[in_offset + 8 * p.in_width],
                                     in_ptr_base[in_offset + 9 * p.in_width],
                                     in_ptr_base[in_offset + 10 * p.in_width],
                                     in_ptr_base[in_offset + 11 * p.in_width]};
                  float32x4_t vi1 = vextq_f32(vi0, vi4, 1);
                  float32x4_t vi2 = vextq_f32(vi0, vi4, 2);
                  float32x4_t vi3 = vextq_f32(vi0, vi4, 3);
                  float32x4_t vi5 = vextq_f32(vi4, vi8, 1);
                  float32x4_t vi6 = vextq_f32(vi4, vi8, 2);

#if defined(__aarch64__)
                  vo0 = vfmaq_laneq_f32(vo0, vi0, vf00, 0);
                vo0 = vfmaq_laneq_f32(vo0, vi1, vf00, 1);
                vo0 = vfmaq_laneq_f32(vo0, vi2, vf00, 2);
                vo0 = vfmaq_laneq_f32(vo0, vi3, vf00, 3);
                vo0 = vfmaq_laneq_f32(vo0, vi4, vf01, 1);
                vo0 = vfmaq_laneq_f32(vo0, vi5, vf01, 2);
                vo0 = vfmaq_laneq_f32(vo0, vi6, vf01, 3);
#else
                  vo0 = vmlaq_lane_f32(vo0, vi0, vget_low_f32(vf00), 0);
                  vo0 = vmlaq_lane_f32(vo0, vi1, vget_low_f32(vf00), 1);
                  vo0 = vmlaq_lane_f32(vo0, vi2, vget_high_f32(vf00), 0);
                  vo0 = vmlaq_lane_f32(vo0, vi3, vget_high_f32(vf00), 1);
                  vo0 = vmlaq_lane_f32(vo0, vi4, vget_low_f32(vf01), 1);
                  vo0 = vmlaq_lane_f32(vo0, vi5, vget_high_f32(vf01), 0);
                  vo0 = vmlaq_lane_f32(vo0, vi6, vget_high_f32(vf01), 1);
#endif

                  out_ptr0_base[out_offset] = vo0[0];
                  out_ptr0_base[out_offset + p.out_width] = vo0[1];
                  out_ptr0_base[out_offset + 2 * p.out_width] = vo0[2];
                  out_ptr0_base[out_offset + 3 * p.out_width] = vo0[3];
                }  // w
              }    // h
            }  // c
          }
        }  // if
      }    // m
    }      // b
  }, 0, p.batch, 1, 0, p.out_channels, 4);

  return MaceStatus::MACE_SUCCESS;
}

template<>
MaceStatus Conv2dK1x15S1<float>::DoCompute(
    const ConvComputeParam &p, const float *filter_data,
    const float *input_data, float *output_data) {
  const index_t tile_height =
      p.out_channels < 4 ? RoundUpDiv4(p.out_height) : p.out_height;

  p.thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        for (index_t h = 0; h < p.out_height; h += tile_height) {
          float *out_ptr_base =
              output_data + b * p.out_batch_size + m * p.out_image_size;
          for (index_t c = 0; c < p.in_channels; ++c) {
            const float *in_ptr_base =
                input_data + b * p.in_batch_size + c * p.in_image_size;
            const float *filter_ptr =
                filter_data + m * p.in_channels * 15 + c * 15;
            /* load filter (1 outch x 4 height x 1 width) */
            float32x4_t vf0, vf1, vf2, vf3;
            vf0 = vld1q_f32(filter_ptr);
            vf1 = vld1q_f32(filter_ptr + 4);
            vf2 = vld1q_f32(filter_ptr + 8);
            vf3 = vld1q_f32(filter_ptr + 11);

            for (index_t ht = 0; ht < tile_height && h + ht < p.out_height;
                 ++ht) {
              for (index_t w = 0; w + 3 < p.out_width; w += 4) {
                // output (1 outch x 1 height x 4 width): vo_outch_height
                float32x4_t vo;
                // load output
                index_t out_offset = (h + ht) * p.out_width + w;
                vo = vld1q_f32(out_ptr_base + out_offset);

                // input (3 slide)
                float32x4_t vi0, vi1, vi2, vi3, vi4, vi5, vi6, vi7, vi8, vi9,
                    vi10, vi11, vi12, vi13, vi14, vi16;
                // input offset
                index_t in_offset = (h + ht) * p.in_width + w;
                // load input
                vi0 = vld1q_f32(in_ptr_base + in_offset);
                vi4 = vld1q_f32(in_ptr_base + in_offset + 4);
                vi8 = vld1q_f32(in_ptr_base + in_offset + 8);
                vi12 = vld1q_f32(in_ptr_base + in_offset + 12);
                vi16 = vld1q_f32(in_ptr_base + in_offset + 16);
                vi1 = vextq_f32(vi0, vi4, 1);
                vi2 = vextq_f32(vi0, vi4, 2);
                vi3 = vextq_f32(vi0, vi4, 3);
                vi5 = vextq_f32(vi4, vi8, 1);
                vi6 = vextq_f32(vi4, vi8, 2);
                vi7 = vextq_f32(vi4, vi8, 3);
                vi9 = vextq_f32(vi8, vi12, 1);
                vi10 = vextq_f32(vi8, vi12, 2);
                vi11 = vextq_f32(vi8, vi12, 3);
                vi13 = vextq_f32(vi12, vi16, 1);
                vi14 = vextq_f32(vi12, vi16, 2);

                vo = vmlaq_lane_f32(vo, vi0, vget_low_f32(vf0), 0);
                vo = vmlaq_lane_f32(vo, vi1, vget_low_f32(vf0), 1);
                vo = vmlaq_lane_f32(vo, vi2, vget_high_f32(vf0), 0);
                vo = vmlaq_lane_f32(vo, vi3, vget_high_f32(vf0), 1);
                vo = vmlaq_lane_f32(vo, vi4, vget_low_f32(vf1), 0);
                vo = vmlaq_lane_f32(vo, vi5, vget_low_f32(vf1), 1);
                vo = vmlaq_lane_f32(vo, vi6, vget_high_f32(vf1), 0);
                vo = vmlaq_lane_f32(vo, vi7, vget_high_f32(vf1), 1);
                vo = vmlaq_lane_f32(vo, vi8, vget_low_f32(vf2), 0);
                vo = vmlaq_lane_f32(vo, vi9, vget_low_f32(vf2), 1);
                vo = vmlaq_lane_f32(vo, vi10, vget_high_f32(vf2), 0);
                vo = vmlaq_lane_f32(vo, vi11, vget_high_f32(vf2), 1);
                vo = vmlaq_lane_f32(vo, vi12, vget_low_f32(vf3), 1);
                vo = vmlaq_lane_f32(vo, vi13, vget_high_f32(vf3), 0);
                vo = vmlaq_lane_f32(vo, vi14, vget_high_f32(vf3), 1);

                vst1q_f32(out_ptr_base + out_offset, vo);
              }  // w
            }    // ht
          }  // c
        }    // h
      }      // m
    }        // b
  }, 0, p.batch, 1, 0, p.out_channels, 1);

  return MaceStatus::MACE_SUCCESS;
}

template<>
MaceStatus Conv2dK15x1S1<float>::DoCompute(
    const ConvComputeParam &p, const float *filter_data,
    const float *input_data, float *output_data) {

  const index_t tile_width =
      p.out_channels < 4 ? RoundUpDiv4(p.out_width) : p.out_width;
  p.thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        for (index_t w = 0; w < p.out_width; w += tile_width) {
          float *out_ptr_base =
              output_data + b * p.out_batch_size + m * p.out_image_size;
          for (index_t c = 0; c < p.in_channels; ++c) {
            const float *in_ptr_base =
                input_data + b * p.in_batch_size + c * p.in_image_size;
            const float *filter_ptr =
                filter_data + m * p.in_channels * 15 + c * 15;
            /* load filter (1 outch x 4 height x 1 width) */
            float32x4_t vf0, vf1, vf2, vf3;
            vf0 = vld1q_f32(filter_ptr);
            vf1 = vld1q_f32(filter_ptr + 4);
            vf2 = vld1q_f32(filter_ptr + 8);
            vf3 = vld1q_f32(filter_ptr + 11);

            for (index_t h = 0; h + 3 < p.out_height; h += 4) {
              for (index_t wt = 0; wt < tile_width && w + wt < p.out_width;
                   ++wt) {
                // load output
                index_t out_offset = h * p.out_width + w + wt;
                // output (1 outch x 4 height x 1 width): vo_outch_height
                float32x4_t vo = {out_ptr_base[out_offset],
                                  out_ptr_base[out_offset + p.out_width],
                                  out_ptr_base[out_offset + 2 * p.out_width],
                                  out_ptr_base[out_offset + 3 * p.out_width]};

                // input offset
                index_t in_offset = h * p.in_width + w + wt;
                // input (3 slide)
                float32x4_t vi0 = {in_ptr_base[in_offset],
                                   in_ptr_base[in_offset + p.in_width],
                                   in_ptr_base[in_offset + 2 * p.in_width],
                                   in_ptr_base[in_offset + 3 * p.in_width]};
                float32x4_t vi4 = {in_ptr_base[in_offset + 4 * p.in_width],
                                   in_ptr_base[in_offset + 5 * p.in_width],
                                   in_ptr_base[in_offset + 6 * p.in_width],
                                   in_ptr_base[in_offset + 7 * p.in_width]};
                float32x4_t vi8 = {in_ptr_base[in_offset + 8 * p.in_width],
                                   in_ptr_base[in_offset + 9 * p.in_width],
                                   in_ptr_base[in_offset + 10 * p.in_width],
                                   in_ptr_base[in_offset + 11 * p.in_width]};
                float32x4_t vi12 = {in_ptr_base[in_offset + 12 * p.in_width],
                                    in_ptr_base[in_offset + 13 * p.in_width],
                                    in_ptr_base[in_offset + 14 * p.in_width],
                                    in_ptr_base[in_offset + 15 * p.in_width]};
                float32x4_t vi16 = {in_ptr_base[in_offset + 16 * p.in_width],
                                    in_ptr_base[in_offset + 17 * p.in_width]};
                float32x4_t vi1 = vextq_f32(vi0, vi4, 1);
                float32x4_t vi2 = vextq_f32(vi0, vi4, 2);
                float32x4_t vi3 = vextq_f32(vi0, vi4, 3);
                float32x4_t vi5 = vextq_f32(vi4, vi8, 1);
                float32x4_t vi6 = vextq_f32(vi4, vi8, 2);
                float32x4_t vi7 = vextq_f32(vi4, vi8, 3);
                float32x4_t vi9 = vextq_f32(vi8, vi12, 1);
                float32x4_t vi10 = vextq_f32(vi8, vi12, 2);
                float32x4_t vi11 = vextq_f32(vi8, vi12, 3);
                float32x4_t vi13 = vextq_f32(vi12, vi16, 1);
                float32x4_t vi14 = vextq_f32(vi12, vi16, 2);

                vo = vmlaq_lane_f32(vo, vi0, vget_low_f32(vf0), 0);
                vo = vmlaq_lane_f32(vo, vi1, vget_low_f32(vf0), 1);
                vo = vmlaq_lane_f32(vo, vi2, vget_high_f32(vf0), 0);
                vo = vmlaq_lane_f32(vo, vi3, vget_high_f32(vf0), 1);
                vo = vmlaq_lane_f32(vo, vi4, vget_low_f32(vf1), 0);
                vo = vmlaq_lane_f32(vo, vi5, vget_low_f32(vf1), 1);
                vo = vmlaq_lane_f32(vo, vi6, vget_high_f32(vf1), 0);
                vo = vmlaq_lane_f32(vo, vi7, vget_high_f32(vf1), 1);
                vo = vmlaq_lane_f32(vo, vi8, vget_low_f32(vf2), 0);
                vo = vmlaq_lane_f32(vo, vi9, vget_low_f32(vf2), 1);
                vo = vmlaq_lane_f32(vo, vi10, vget_high_f32(vf2), 0);
                vo = vmlaq_lane_f32(vo, vi11, vget_high_f32(vf2), 1);
                vo = vmlaq_lane_f32(vo, vi12, vget_low_f32(vf3), 1);
                vo = vmlaq_lane_f32(vo, vi13, vget_high_f32(vf3), 0);
                vo = vmlaq_lane_f32(vo, vi14, vget_high_f32(vf3), 1);

                out_ptr_base[out_offset] = vo[0];
                out_ptr_base[out_offset + p.out_width] = vo[1];
                out_ptr_base[out_offset + 2 * p.out_width] = vo[2];
                out_ptr_base[out_offset + 3 * p.out_width] = vo[3];
              }  // wt
            }    // h
          }  // c
        }    // w
      }      // m
    }        // b
  }, 0, p.batch, 1, 0, p.out_channels, 1);

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
