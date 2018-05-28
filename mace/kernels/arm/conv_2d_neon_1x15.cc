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
#include "mace/utils/logging.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

inline void Conv2dCPUK1x15Calc(const float *in_ptr,
                               const float *filter_ptr,
                               const index_t in_width,
                               const index_t in_channels,
                               const index_t out_height,
                               const index_t h,
                               const index_t tile_height,
                               const index_t out_width,
                               const index_t out_image_size,
                               float *out_ptr,
                               const index_t io,
                               const int stride) {
  for (index_t ih = 0; ih < tile_height && h + ih < out_height; ++ih) {
    for (index_t iw = 0; iw < out_width; ++iw) {
      for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 15; ++j) {
          out_ptr[io * out_image_size + (h + ih) * out_width + iw] +=
              in_ptr[((h + ih) * stride + i) * in_width + (iw * stride + j)] *
              filter_ptr[io * in_channels * 15 + i * 15 + j];
        }
      }
    }
  }
}

// Ho = 1, Wo = 4, Co = 1
void Conv2dNeonK1x15S1(const float *input,
                       const float *filter,
                       const index_t *in_shape,
                       const index_t *out_shape,
                       float *output) {
  const index_t in_image_size = in_shape[2] * in_shape[3];
  const index_t out_image_size = out_shape[2] * out_shape[3];
  const index_t in_batch_size = in_shape[1] * in_image_size;
  const index_t out_batch_size = out_shape[1] * out_image_size;
  const index_t tile_height =
      out_shape[1] < 4 ? RoundUpDiv4(out_shape[2]) : out_shape[2];

#pragma omp parallel for collapse(3)
  for (index_t b = 0; b < out_shape[0]; ++b) {
    for (index_t m = 0; m < out_shape[1]; ++m) {
      for (index_t h = 0; h < out_shape[2]; h += tile_height) {
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

          for (index_t ht = 0; ht < tile_height && h + ht < out_height; ++ht) {
            for (index_t w = 0; w + 3 < out_width; w += 4) {
              // output (1 outch x 1 height x 4 width): vo_outch_height
              float32x4_t vo;
              // load output
              index_t out_offset = (h + ht) * out_width + w;
              vo = vld1q_f32(out_ptr_base + out_offset);

              // input (3 slide)
              float32x4_t vi0, vi1, vi2, vi3, vi4, vi5, vi6, vi7, vi8, vi9,
                  vi10, vi11, vi12, vi13, vi14, vi16;
              // input offset
              index_t in_offset = (h + ht) * in_width + w;
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
#else
          Conv2dCPUK1x15Calc(in_ptr_base, filter_ptr, in_width, in_channels,
                             out_height, h, tile_height, out_width,
                             out_image_size, out_ptr_base, 0, 1);
#endif
        }  // c
      }    // h
    }      // m
  }        // b
}

}  // namespace kernels
}  // namespace mace
