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
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

inline void Conv2dCPUK15x1Calc(const float *in_ptr,
                               const float *filter_ptr,
                               const index_t in_width,
                               const index_t in_channels,
                               const index_t out_height,
                               const index_t out_width,
                               const index_t w,
                               const index_t tile_width,
                               const index_t out_image_size,
                               float *out_ptr,
                               const index_t io,
                               const int stride) {
  for (index_t ih = 0; ih < out_height; ++ih) {
    for (index_t iw = 0; iw < tile_width && w + iw < out_width; ++iw) {
      for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 1; ++j) {
          out_ptr[io * out_image_size + ih * out_width + w + iw] +=
              in_ptr[(ih * stride + i) * in_width + ((w + iw) * stride + j)] *
              filter_ptr[io * in_channels * 15 + i * 1 + j];
        }
      }
    }
  }
}

// Ho = 4, Wo = 1, Co = 1
void Conv2dNeonK15x1S1(const float *input,
                       const float *filter,
                       const index_t *in_shape,
                       const index_t *out_shape,
                       float *output) {
  const index_t in_image_size = in_shape[2] * in_shape[3];
  const index_t out_image_size = out_shape[2] * out_shape[3];
  const index_t in_batch_size = in_shape[1] * in_image_size;
  const index_t out_batch_size = out_shape[1] * out_image_size;
  const index_t tile_width =
      out_shape[1] < 4 ? RoundUpDiv4(out_shape[3]) : out_shape[3];

#pragma omp parallel for collapse(3)
  for (index_t b = 0; b < out_shape[0]; ++b) {
    for (index_t m = 0; m < out_shape[1]; ++m) {
      for (index_t w = 0; w < out_shape[3]; w += tile_width) {
        const index_t out_height = out_shape[2];
        const index_t out_width = out_shape[3];
        const index_t in_channels = in_shape[1];
        const index_t in_width = in_shape[3];
        float *out_ptr_base = output + b * out_batch_size + m * out_image_size;
        for (index_t c = 0; c < in_channels; ++c) {
          const float *in_ptr_base =
              input + b * in_batch_size + c * in_image_size;
          const float *filter_ptr = filter + m * in_channels * 15 + c * 15;
#if defined(MACE_ENABLE_NEON) && !defined(__aarch64__)
          /* load filter (1 outch x 4 height x 1 width) */
          float32x4_t vf0, vf1, vf2, vf3;
          vf0 = vld1q_f32(filter_ptr);
          vf1 = vld1q_f32(filter_ptr + 4);
          vf2 = vld1q_f32(filter_ptr + 8);
          vf3 = vld1q_f32(filter_ptr + 11);

          for (index_t h = 0; h + 3 < out_height; h += 4) {
            for (index_t wt = 0; wt < tile_width && w + wt < out_width; ++wt) {
              // load output
              index_t out_offset = h * out_width + w + wt;
              // output (1 outch x 4 height x 1 width): vo_outch_height
              float32x4_t vo = {out_ptr_base[out_offset],
                                out_ptr_base[out_offset + out_width],
                                out_ptr_base[out_offset + 2 * out_width],
                                out_ptr_base[out_offset + 3 * out_width]};

              // input offset
              index_t in_offset = h * in_width + w + wt;
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
              float32x4_t vi12 = {in_ptr_base[in_offset + 12 * in_width],
                                  in_ptr_base[in_offset + 13 * in_width],
                                  in_ptr_base[in_offset + 14 * in_width],
                                  in_ptr_base[in_offset + 15 * in_width]};
              float32x4_t vi16 = {in_ptr_base[in_offset + 16 * in_width],
                                  in_ptr_base[in_offset + 17 * in_width]};
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
              out_ptr_base[out_offset + out_width] = vo[1];
              out_ptr_base[out_offset + 2 * out_width] = vo[2];
              out_ptr_base[out_offset + 3 * out_width] = vo[3];
            }  // wt
          }    // h
#else
          Conv2dCPUK15x1Calc(in_ptr_base, filter_ptr, in_width, in_channels,
                             out_height, out_width, w, tile_width,
                             out_image_size, out_ptr_base, 0, 1);
#endif
        }  // c
      }    // w
    }      // m
  }        // b
}

}  // namespace kernels
}  // namespace mace
