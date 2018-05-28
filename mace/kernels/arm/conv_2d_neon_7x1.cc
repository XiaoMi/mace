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

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#include "mace/kernels/arm/conv_2d_neon.h"

namespace mace {
namespace kernels {

// Ho = 4, Wo = 1, Co = 4
void Conv2dNeonK7x1S1(const float *input,
                      const float *filter,
                      const index_t *in_shape,
                      const index_t *out_shape,
                      float *output) {
  const index_t in_image_size = in_shape[2] * in_shape[3];
  const index_t out_image_size = out_shape[2] * out_shape[3];
  const index_t in_batch_size = in_shape[1] * in_image_size;
  const index_t out_batch_size = out_shape[1] * out_image_size;

#pragma omp parallel for collapse(2)
  for (index_t b = 0; b < out_shape[0]; ++b) {
    for (index_t m = 0; m < out_shape[1]; m += 4) {
      const index_t out_channels = out_shape[1];
      const index_t out_height = out_shape[2];
      const index_t out_width = out_shape[3];
      const index_t in_channels = in_shape[1];
      const index_t in_width = in_shape[3];
      if (m + 3 < out_channels) {
        float *out_ptr0_base = output + b * out_batch_size + m * out_image_size;
#if defined(MACE_ENABLE_NEON)
        float *out_ptr1_base =
            output + b * out_batch_size + (m + 1) * out_image_size;
        float *out_ptr2_base =
            output + b * out_batch_size + (m + 2) * out_image_size;
        float *out_ptr3_base =
            output + b * out_batch_size + (m + 3) * out_image_size;
#endif
        for (index_t c = 0; c < in_channels; ++c) {
          const float *in_ptr_base =
              input + b * in_batch_size + c * in_image_size;
          const float *filter_ptr0 = filter + m * in_channels * 7 + c * 7;
#if defined(MACE_ENABLE_NEON)
          const float *filter_ptr1 = filter + (m + 1) * in_channels * 7 + c * 7;
          const float *filter_ptr2 = filter + (m + 2) * in_channels * 7 + c * 7;
          const float *filter_ptr3 = filter + (m + 3) * in_channels * 7 + c * 7;
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

          for (index_t h = 0; h + 3 < out_height; h += 4) {
            for (index_t w = 0; w < out_width; ++w) {
              // load output
              index_t out_offset = h * out_width + w;
              // output (4 outch x 4 height x 1 width): vo_outch_height
              float32x4_t vo0 = {out_ptr0_base[out_offset],
                                 out_ptr0_base[out_offset + out_width],
                                 out_ptr0_base[out_offset + 2 * out_width],
                                 out_ptr0_base[out_offset + 3 * out_width]};
              float32x4_t vo1 = {out_ptr1_base[out_offset],
                                 out_ptr1_base[out_offset + out_width],
                                 out_ptr1_base[out_offset + 2 * out_width],
                                 out_ptr1_base[out_offset + 3 * out_width]};
              float32x4_t vo2 = {out_ptr2_base[out_offset],
                                 out_ptr2_base[out_offset + out_width],
                                 out_ptr2_base[out_offset + 2 * out_width],
                                 out_ptr2_base[out_offset + 3 * out_width]};
              float32x4_t vo3 = {out_ptr3_base[out_offset],
                                 out_ptr3_base[out_offset + out_width],
                                 out_ptr3_base[out_offset + 2 * out_width],
                                 out_ptr3_base[out_offset + 3 * out_width]};

              // input offset
              index_t in_offset = h * in_width + w;
              // input (3 slide)
              float32x4_t vi0 = {in_ptr_base[in_offset],
                                 in_ptr_base[in_offset + in_width],
                                 in_ptr_base[in_offset + 2 * in_width],
                                 in_ptr_base[in_offset + 3 * in_width]};
              float32x4_t vi4 = {in_ptr_base[in_offset + 4 * in_width],
                                 in_ptr_base[in_offset + 5 * in_width],
                                 in_ptr_base[in_offset + 6 * in_width],
                                 in_ptr_base[in_offset + 7 * in_width]};
              float32x4_t vi8 = {in_ptr_base[in_offset + 8 * in_width],
                                 in_ptr_base[in_offset + 9 * in_width]};
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
              out_ptr0_base[out_offset + out_width] = vo0[1];
              out_ptr0_base[out_offset + 2 * out_width] = vo0[2];
              out_ptr0_base[out_offset + 3 * out_width] = vo0[3];
              out_ptr1_base[out_offset] = vo1[0];
              out_ptr1_base[out_offset + out_width] = vo1[1];
              out_ptr1_base[out_offset + 2 * out_width] = vo1[2];
              out_ptr1_base[out_offset + 3 * out_width] = vo1[3];
              out_ptr2_base[out_offset] = vo2[0];
              out_ptr2_base[out_offset + out_width] = vo2[1];
              out_ptr2_base[out_offset + 2 * out_width] = vo2[2];
              out_ptr2_base[out_offset + 3 * out_width] = vo2[3];
              out_ptr3_base[out_offset] = vo3[0];
              out_ptr3_base[out_offset + out_width] = vo3[1];
              out_ptr3_base[out_offset + 2 * out_width] = vo3[2];
              out_ptr3_base[out_offset + 3 * out_width] = vo3[3];
            }  // w
          }    // h
#else
          for (index_t oc = 0; oc < 4; ++oc) {
            Conv2dCPUKHxKWCalc(in_ptr_base, filter_ptr0 + oc * in_channels * 7,
                               in_width, 7, 1, out_height, out_width,
                               out_ptr0_base + oc * out_image_size, 1);
          }
#endif
        }  // c
      } else {
        for (index_t mm = m; mm < out_channels; ++mm) {
          float *out_ptr0_base =
              output + b * out_batch_size + mm * out_image_size;
          for (index_t c = 0; c < in_channels; ++c) {
            const float *in_ptr_base =
                input + b * in_batch_size + c * in_image_size;
            const float *filter_ptr0 = filter + mm * in_channels * 7 + c * 7;
#if defined(MACE_ENABLE_NEON)
            /* load filter (1 outch x 4 height x 1 width) */
            float32x4_t vf00, vf01;
            vf00 = vld1q_f32(filter_ptr0);
            vf01 = vld1q_f32(filter_ptr0 + 3);

            for (index_t h = 0; h + 3 < out_height; h += 4) {
              for (index_t w = 0; w < out_width; ++w) {
                // load output
                index_t out_offset = h * out_width + w;
                // output (1 outch x 4 height x 1 width): vo_outch_height
                float32x4_t vo0 = {out_ptr0_base[out_offset],
                                   out_ptr0_base[out_offset + out_width],
                                   out_ptr0_base[out_offset + 2 * out_width],
                                   out_ptr0_base[out_offset + 3 * out_width]};

                // input offset
                index_t in_offset = h * in_width + w;
                // input (3 slide)
                float32x4_t vi0 = {in_ptr_base[in_offset],
                                   in_ptr_base[in_offset + in_width],
                                   in_ptr_base[in_offset + 2 * in_width],
                                   in_ptr_base[in_offset + 3 * in_width]};
                float32x4_t vi4 = {in_ptr_base[in_offset + 4 * in_width],
                                   in_ptr_base[in_offset + 5 * in_width],
                                   in_ptr_base[in_offset + 6 * in_width],
                                   in_ptr_base[in_offset + 7 * in_width]};
                float32x4_t vi8 = {in_ptr_base[in_offset + 8 * in_width],
                                   in_ptr_base[in_offset + 9 * in_width],
                                   in_ptr_base[in_offset + 10 * in_width],
                                   in_ptr_base[in_offset + 11 * in_width]};
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
                out_ptr0_base[out_offset + out_width] = vo0[1];
                out_ptr0_base[out_offset + 2 * out_width] = vo0[2];
                out_ptr0_base[out_offset + 3 * out_width] = vo0[3];
              }  // w
            }    // h
#else
            Conv2dCPUKHxKWCalc(in_ptr_base, filter_ptr0, in_width, 7, 1,
                               out_height, out_width, out_ptr0_base, 1);
#endif
          }  // c
        }
      }  // if
    }    // m
  }      // b
}

}  // namespace kernels
}  // namespace mace
