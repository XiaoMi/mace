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

#include "mace/ops/arm/base/conv_2d_3x3.h"

#include <arm_neon.h>
#include <memory>
#include <string>

#include "mace/ops/arm/base/common_neon.h"
#include "mace/ops/delegator/conv_2d.h"

namespace mace {
namespace ops {
namespace arm {

template <>
MaceStatus Conv2dK3x3S1<BFloat16>::DoCompute(
    const ConvComputeParam &p, const BFloat16 *filter_data,
    const BFloat16 *input_data, BFloat16 *output_data) {
  p.thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        if (m + 1 < p.out_channels) {
          auto in_ptr0_base = input_data + b * p.in_batch_size;
          auto in_ptr1_base = in_ptr0_base + p.in_width;
          auto in_ptr2_base = in_ptr1_base + p.in_width;
          auto in_ptr3_base = in_ptr2_base + p.in_width;
          auto out_ptr0 = output_data + b * p.out_batch_size
              + m * p.out_image_size;
          auto out_ptr1 = out_ptr0 + p.out_image_size;
          for (index_t h = 0; h + 1 < p.out_height; h += 2) {
            for (index_t w = 0; w + 3 < p.out_width; w += 4) {
              auto in_ptr0 = in_ptr0_base;
              auto in_ptr1 = in_ptr1_base;
              auto in_ptr2 = in_ptr2_base;
              auto in_ptr3 = in_ptr3_base;
              auto filter_ptr0 = filter_data + m * p.in_channels * 9;
              auto filter_ptr1 = filter_ptr0 + p.in_channels * 9;
              float32x4_t vo00 = vdupq_n_f32(0.f);
              float32x4_t vo01 = vdupq_n_f32(0.f);
              float32x4_t vo10 = vdupq_n_f32(0.f);
              float32x4_t vo11 = vdupq_n_f32(0.f);
              for (index_t c = 0; c < p.in_channels; ++c) {
                // input (4 height x 3 slide): vi_height_slide
                float32x4_t vi00, vi01, vi02, vi0n;
                float32x4_t vi10, vi11, vi12, vi1n;
                float32x4_t vi20, vi21, vi22, vi2n;
                float32x4_t vi30, vi31, vi32, vi3n;

                // load input
                vi00 = vld1q_bf16(in_ptr0);
                vi0n = vld1q_bf16(in_ptr0 + 4);
                vi10 = vld1q_bf16(in_ptr1);
                vi1n = vld1q_bf16(in_ptr1 + 4);
                vi20 = vld1q_bf16(in_ptr2);
                vi2n = vld1q_bf16(in_ptr2 + 4);
                vi30 = vld1q_bf16(in_ptr3);
                vi3n = vld1q_bf16(in_ptr3 + 4);

                vi01 = vextq_f32(vi00, vi0n, 1);
                vi02 = vextq_f32(vi00, vi0n, 2);
                vi11 = vextq_f32(vi10, vi1n, 1);
                vi12 = vextq_f32(vi10, vi1n, 2);
                vi21 = vextq_f32(vi20, vi2n, 1);
                vi22 = vextq_f32(vi20, vi2n, 2);
                vi31 = vextq_f32(vi30, vi3n, 1);
                vi32 = vextq_f32(vi30, vi3n, 2);

#if defined(__aarch64__)
                // load filter (2 outch x 3 height x 3 width):
                // vf_outch_height
                float32x4_t vf00, vf01, vf02;
                float32x4_t vf10, vf11, vf12;
                vf00 = vld1q_bf16(filter_ptr0);
                vf01 = vld1q_bf16(filter_ptr0 + 3);
                vf02 = vld1q_bf16(filter_ptr0 + 6);
                vf10 = vld1q_bf16(filter_ptr1);
                vf11 = vld1q_bf16(filter_ptr1 + 3);
                vf12 = vld1q_bf16(filter_ptr1 + 6);

                // outch 0, height 0
                vo00 = vfmaq_laneq_f32(vo00, vi00, vf00, 0);  // reg count: 18
                vo00 = vfmaq_laneq_f32(vo00, vi01, vf00, 1);
                vo00 = vfmaq_laneq_f32(vo00, vi02, vf00, 2);
                vo00 = vfmaq_laneq_f32(vo00, vi10, vf01, 0);
                vo00 = vfmaq_laneq_f32(vo00, vi11, vf01, 1);
                vo00 = vfmaq_laneq_f32(vo00, vi12, vf01, 2);
                vo00 = vfmaq_laneq_f32(vo00, vi20, vf02, 0);
                vo00 = vfmaq_laneq_f32(vo00, vi21, vf02, 1);
                vo00 = vfmaq_laneq_f32(vo00, vi22, vf02, 2);

                // outch 0, height 1
                vo01 = vfmaq_laneq_f32(vo01, vi10, vf00, 0);
                vo01 = vfmaq_laneq_f32(vo01, vi11, vf00, 1);
                vo01 = vfmaq_laneq_f32(vo01, vi12, vf00, 2);
                vo01 = vfmaq_laneq_f32(vo01, vi20, vf01, 0);
                vo01 = vfmaq_laneq_f32(vo01, vi21, vf01, 1);
                vo01 = vfmaq_laneq_f32(vo01, vi22, vf01, 2);
                vo01 = vfmaq_laneq_f32(vo01, vi30, vf02, 0);
                vo01 = vfmaq_laneq_f32(vo01, vi31, vf02, 1);
                vo01 = vfmaq_laneq_f32(vo01, vi32, vf02, 2);

                // outch 1, height 0
                vo10 = vfmaq_laneq_f32(vo10, vi00, vf10, 0);
                vo10 = vfmaq_laneq_f32(vo10, vi01, vf10, 1);
                vo10 = vfmaq_laneq_f32(vo10, vi02, vf10, 2);
                vo10 = vfmaq_laneq_f32(vo10, vi10, vf11, 0);
                vo10 = vfmaq_laneq_f32(vo10, vi11, vf11, 1);
                vo10 = vfmaq_laneq_f32(vo10, vi12, vf11, 2);
                vo10 = vfmaq_laneq_f32(vo10, vi20, vf12, 0);
                vo10 = vfmaq_laneq_f32(vo10, vi21, vf12, 1);
                vo10 = vfmaq_laneq_f32(vo10, vi22, vf12, 2);

                // outch 1, height 1
                vo11 = vfmaq_laneq_f32(vo11, vi10, vf10, 0);
                vo11 = vfmaq_laneq_f32(vo11, vi11, vf10, 1);
                vo11 = vfmaq_laneq_f32(vo11, vi12, vf10, 2);
                vo11 = vfmaq_laneq_f32(vo11, vi20, vf11, 0);
                vo11 = vfmaq_laneq_f32(vo11, vi21, vf11, 1);
                vo11 = vfmaq_laneq_f32(vo11, vi22, vf11, 2);
                vo11 = vfmaq_laneq_f32(vo11, vi30, vf12, 0);
                vo11 = vfmaq_laneq_f32(vo11, vi31, vf12, 1);
                vo11 = vfmaq_laneq_f32(vo11, vi32, vf12, 2);
#else
                float32x2_t vf001, vf023, vf045, vf067, vf089;
                float32x2_t vf101, vf123, vf145, vf167, vf189;
                vf001 = vld1_bf16(filter_ptr0);
                vf023 = vld1_bf16(filter_ptr0 + 2);
                vf045 = vld1_bf16(filter_ptr0 + 4);
                vf067 = vld1_bf16(filter_ptr0 + 6);
                vf089 = vld1_bf16(filter_ptr0 + 8);

                vf101 = vld1_bf16(filter_ptr1);
                vf123 = vld1_bf16(filter_ptr1 + 2);
                vf145 = vld1_bf16(filter_ptr1 + 4);
                vf167 = vld1_bf16(filter_ptr1 + 6);
                vf189 = vld1_bf16(filter_ptr1 + 8);

                // outch 0, height 0
                vo00 = vmlaq_lane_f32(vo00, vi00, vf001, 0);
                vo00 = vmlaq_lane_f32(vo00, vi01, vf001, 1);
                vo00 = vmlaq_lane_f32(vo00, vi02, vf023, 0);
                vo00 = vmlaq_lane_f32(vo00, vi10, vf023, 1);
                vo00 = vmlaq_lane_f32(vo00, vi11, vf045, 0);
                vo00 = vmlaq_lane_f32(vo00, vi12, vf045, 1);
                vo00 = vmlaq_lane_f32(vo00, vi20, vf067, 0);
                vo00 = vmlaq_lane_f32(vo00, vi21, vf067, 1);
                vo00 = vmlaq_lane_f32(vo00, vi22, vf089, 0);

                // outch 0, height 1
                vo01 = vmlaq_lane_f32(vo01, vi10, vf001, 0);
                vo01 = vmlaq_lane_f32(vo01, vi11, vf001, 1);
                vo01 = vmlaq_lane_f32(vo01, vi12, vf023, 0);
                vo01 = vmlaq_lane_f32(vo01, vi20, vf023, 1);
                vo01 = vmlaq_lane_f32(vo01, vi21, vf045, 0);
                vo01 = vmlaq_lane_f32(vo01, vi22, vf045, 1);
                vo01 = vmlaq_lane_f32(vo01, vi30, vf067, 0);
                vo01 = vmlaq_lane_f32(vo01, vi31, vf067, 1);
                vo01 = vmlaq_lane_f32(vo01, vi32, vf089, 0);

                // outch 1, height 0
                vo10 = vmlaq_lane_f32(vo10, vi00, vf101, 0);
                vo10 = vmlaq_lane_f32(vo10, vi01, vf101, 1);
                vo10 = vmlaq_lane_f32(vo10, vi02, vf123, 0);
                vo10 = vmlaq_lane_f32(vo10, vi10, vf123, 1);
                vo10 = vmlaq_lane_f32(vo10, vi11, vf145, 0);
                vo10 = vmlaq_lane_f32(vo10, vi12, vf145, 1);
                vo10 = vmlaq_lane_f32(vo10, vi20, vf167, 0);
                vo10 = vmlaq_lane_f32(vo10, vi21, vf167, 1);
                vo10 = vmlaq_lane_f32(vo10, vi22, vf189, 0);

                // outch 1, height 1
                vo11 = vmlaq_lane_f32(vo11, vi10, vf101, 0);
                vo11 = vmlaq_lane_f32(vo11, vi11, vf101, 1);
                vo11 = vmlaq_lane_f32(vo11, vi12, vf123, 0);
                vo11 = vmlaq_lane_f32(vo11, vi20, vf123, 1);
                vo11 = vmlaq_lane_f32(vo11, vi21, vf145, 0);
                vo11 = vmlaq_lane_f32(vo11, vi22, vf145, 1);
                vo11 = vmlaq_lane_f32(vo11, vi30, vf167, 0);
                vo11 = vmlaq_lane_f32(vo11, vi31, vf167, 1);
                vo11 = vmlaq_lane_f32(vo11, vi32, vf189, 0);
#endif
                in_ptr0 += p.in_image_size;
                in_ptr1 += p.in_image_size;
                in_ptr2 += p.in_image_size;
                in_ptr3 += p.in_image_size;
                filter_ptr0 += 9;
                filter_ptr1 += 9;
              }
              vst1q_bf16(out_ptr0, vo00);
              vst1q_bf16(out_ptr0 + p.out_width, vo01);
              vst1q_bf16(out_ptr1, vo10);
              vst1q_bf16(out_ptr1 + p.out_width, vo11);

              in_ptr0_base += 4;
              in_ptr1_base += 4;
              in_ptr2_base += 4;
              in_ptr3_base += 4;

              out_ptr0 += 4;
              out_ptr1 += 4;
            }
            in_ptr0_base += 2 + p.in_width;
            in_ptr1_base += 2 + p.in_width;
            in_ptr2_base += 2 + p.in_width;
            in_ptr3_base += 2 + p.in_width;

            out_ptr0 += p.out_width;
            out_ptr1 += p.out_width;
          }
        } else {
          for (index_t mm = m; mm < p.out_channels; ++mm) {
            auto out_ptr0 = output_data + b * p.out_batch_size +
                mm * p.out_image_size;
            auto in_ptr0_base = input_data + b * p.in_batch_size;
            auto in_ptr1_base = in_ptr0_base + p.in_width;
            auto in_ptr2_base = in_ptr1_base + p.in_width;
            auto in_ptr3_base = in_ptr2_base + p.in_width;
            for (index_t h = 0; h + 1 < p.out_height; h += 2) {
              for (index_t w = 0; w + 3 < p.out_width; w += 4) {
                auto in_ptr0 = in_ptr0_base;
                auto in_ptr1 = in_ptr1_base;
                auto in_ptr2 = in_ptr2_base;
                auto in_ptr3 = in_ptr3_base;
                auto filter_ptr0 = filter_data + mm * p.in_channels * 9;
                float32x4_t vo00 = vdupq_n_f32(0.f);
                float32x4_t vo01 = vdupq_n_f32(0.f);
                for (index_t c = 0; c < p.in_channels; ++c) {
                  // input (4 height x 3 slide): vi_height_slide
                  float32x4_t vi00, vi01, vi02, vi0n;
                  float32x4_t vi10, vi11, vi12, vi1n;
                  float32x4_t vi20, vi21, vi22, vi2n;
                  float32x4_t vi30, vi31, vi32, vi3n;

                  // load input
                  vi00 = vld1q_bf16(in_ptr0);
                  vi0n = vld1q_bf16(in_ptr0 + 4);
                  vi10 = vld1q_bf16(in_ptr1);
                  vi1n = vld1q_bf16(in_ptr1 + 4);
                  vi20 = vld1q_bf16(in_ptr2);
                  vi2n = vld1q_bf16(in_ptr2 + 4);
                  vi30 = vld1q_bf16(in_ptr3);
                  vi3n = vld1q_bf16(in_ptr3 + 4);

                  vi01 = vextq_f32(vi00, vi0n, 1);
                  vi02 = vextq_f32(vi00, vi0n, 2);
                  vi11 = vextq_f32(vi10, vi1n, 1);
                  vi12 = vextq_f32(vi10, vi1n, 2);
                  vi21 = vextq_f32(vi20, vi2n, 1);
                  vi22 = vextq_f32(vi20, vi2n, 2);
                  vi31 = vextq_f32(vi30, vi3n, 1);
                  vi32 = vextq_f32(vi30, vi3n, 2);

#if defined(__aarch64__)
                  // load filter (1 outch x 3 height x 3 width): vf_outch_height
                  float32x4_t vf00, vf01, vf02;
                  vf00 = vld1q_bf16(filter_ptr0);
                  vf01 = vld1q_bf16(filter_ptr0 + 3);
                  vf02 = vld1q_bf16(filter_ptr0 + 5);

                  // outch 0, height 0
                  vo00 = vfmaq_laneq_f32(vo00, vi00, vf00, 0);
                  vo00 = vfmaq_laneq_f32(vo00, vi01, vf00, 1);
                  vo00 = vfmaq_laneq_f32(vo00, vi02, vf00, 2);
                  vo00 = vfmaq_laneq_f32(vo00, vi10, vf01, 0);
                  vo00 = vfmaq_laneq_f32(vo00, vi11, vf01, 1);
                  vo00 = vfmaq_laneq_f32(vo00, vi12, vf01, 2);
                  vo00 = vfmaq_laneq_f32(vo00, vi20, vf02, 1);
                  vo00 = vfmaq_laneq_f32(vo00, vi21, vf02, 2);
                  vo00 = vfmaq_laneq_f32(vo00, vi22, vf02, 3);

                  // outch 0, height 1
                  vo01 = vfmaq_laneq_f32(vo01, vi10, vf00, 0);
                  vo01 = vfmaq_laneq_f32(vo01, vi11, vf00, 1);
                  vo01 = vfmaq_laneq_f32(vo01, vi12, vf00, 2);
                  vo01 = vfmaq_laneq_f32(vo01, vi20, vf01, 0);
                  vo01 = vfmaq_laneq_f32(vo01, vi21, vf01, 1);
                  vo01 = vfmaq_laneq_f32(vo01, vi22, vf01, 2);
                  vo01 = vfmaq_laneq_f32(vo01, vi30, vf02, 1);
                  vo01 = vfmaq_laneq_f32(vo01, vi31, vf02, 2);
                  vo01 = vfmaq_laneq_f32(vo01, vi32, vf02, 3);
#else
                  // load filter (1 outch x 3 height x 3 width): vf_outch_height
                  float32x2_t vf01, vf23, vf45, vf67, vf78;
                  vf01 = vld1_bf16(filter_ptr0);
                  vf23 = vld1_bf16(filter_ptr0 + 2);
                  vf45 = vld1_bf16(filter_ptr0 + 4);
                  vf67 = vld1_bf16(filter_ptr0 + 6);
                  vf78 = vld1_bf16(filter_ptr0 + 7);

                  // outch 0, height 0
                  vo00 = vmlaq_lane_f32(vo00, vi00, vf01, 0);
                  vo00 = vmlaq_lane_f32(vo00, vi01, vf01, 1);
                  vo00 = vmlaq_lane_f32(vo00, vi02, vf23, 0);
                  vo00 = vmlaq_lane_f32(vo00, vi10, vf23, 1);
                  vo00 = vmlaq_lane_f32(vo00, vi11, vf45, 0);
                  vo00 = vmlaq_lane_f32(vo00, vi12, vf45, 1);
                  vo00 = vmlaq_lane_f32(vo00, vi20, vf67, 0);
                  vo00 = vmlaq_lane_f32(vo00, vi21, vf67, 1);
                  vo00 = vmlaq_lane_f32(vo00, vi22, vf78, 1);

                  // outch 0, height 1
                  vo01 = vmlaq_lane_f32(vo01, vi10, vf01, 0);
                  vo01 = vmlaq_lane_f32(vo01, vi11, vf01, 1);
                  vo01 = vmlaq_lane_f32(vo01, vi12, vf23, 0);
                  vo01 = vmlaq_lane_f32(vo01, vi20, vf23, 1);
                  vo01 = vmlaq_lane_f32(vo01, vi21, vf45, 0);
                  vo01 = vmlaq_lane_f32(vo01, vi22, vf45, 1);
                  vo01 = vmlaq_lane_f32(vo01, vi30, vf67, 0);
                  vo01 = vmlaq_lane_f32(vo01, vi31, vf67, 1);
                  vo01 = vmlaq_lane_f32(vo01, vi32, vf78, 1);

#endif
                  in_ptr0 += p.in_image_size;
                  in_ptr1 += p.in_image_size;
                  in_ptr2 += p.in_image_size;
                  in_ptr3 += p.in_image_size;
                  filter_ptr0 += 9;
                }  // c

                vst1q_bf16(out_ptr0, vo00);
                vst1q_bf16(out_ptr0 + p.out_width, vo01);

                in_ptr0_base += 4;
                in_ptr1_base += 4;
                in_ptr2_base += 4;
                in_ptr3_base += 4;

                out_ptr0 += 4;
              }  // w

              in_ptr0_base += 2 + p.in_width;
              in_ptr1_base += 2 + p.in_width;
              in_ptr2_base += 2 + p.in_width;
              in_ptr3_base += 2 + p.in_width;

              out_ptr0 += p.out_width;
            }  // h
          }    // mm
        }      // if
      }        // m
    }          // b
  }, 0, p.batch, 1, 0, p.out_channels, 2);

  return MaceStatus::MACE_SUCCESS;
}

template <>
MaceStatus Conv2dK3x3S2<BFloat16>::DoCompute(
    const ConvComputeParam &p, const BFloat16 *filter_data,
    const BFloat16 *input_data, BFloat16 *output_data) {
  p.thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t m = start1; m < end1; m += step1) {
        auto out_base = output_data + b * p.out_batch_size +
            m * p.out_image_size;
        for (index_t h = 0; h < p.out_height; ++h) {
          for (index_t w = 0; w + 3 < p.out_width; w += 4) {
            // offset
            const index_t in_h = h * 2;
            const index_t in_w = w * 2;
            const index_t in_offset = in_h * p.in_width + in_w;
            const index_t out_offset = h * p.out_width + w;
            // output (1 outch x 1 height x 4 width): vo
            float32x4_t vo = vdupq_n_f32(0.f);
            auto in_base = input_data + b * p.in_batch_size;
            auto f_ptr = filter_data + m * p.in_channels * 9;
            for (index_t c = 0; c < p.in_channels; ++c) {
              // input (3 height x 3 slide): vi_height_slide
              float32x4x2_t vi0, vi1, vi2;
              float32x4_t vi0n, vi1n, vi2n;
              float32x4_t vi00, vi01, vi02;
              float32x4_t vi10, vi11, vi12;
              float32x4_t vi20, vi21, vi22;

              // load input
              vi0 = vld2q_bf16(in_base + in_offset);
              vi1 = vld2q_bf16(in_base + in_offset + p.in_width);
              vi2 = vld2q_bf16(in_base + in_offset + 2 * p.in_width);

              vi0n = vld1q_bf16(in_base + in_offset + 8);
              vi1n = vld1q_bf16(in_base + in_offset + p.in_width + 8);
              vi2n = vld1q_bf16(in_base + in_offset + 2 * p.in_width + 8);

              vi00 = vi0.val[0];                // [0.2.4.6]
              vi01 = vi0.val[1];                // [1.3.5.7]
              vi02 = vextq_f32(vi00, vi0n, 1);  // [2.4.6.8]
              vi10 = vi1.val[0];
              vi11 = vi1.val[1];
              vi12 = vextq_f32(vi10, vi1n, 1);
              vi20 = vi2.val[0];
              vi21 = vi2.val[1];
              vi22 = vextq_f32(vi20, vi2n, 1);

#if defined(__aarch64__)    // arm v8
              // load filter (1 outch x 3 height x 3 width): vf_outch_height
              float32x4_t vf00 = vld1q_bf16(f_ptr);
              float32x4_t vf01 = vld1q_bf16(f_ptr + 3);
              float32x4_t vf02 = vld1q_bf16(f_ptr + 5);

              // outch 0, height 0
              vo = vfmaq_laneq_f32(vo, vi00, vf00, 0);
              vo = vfmaq_laneq_f32(vo, vi01, vf00, 1);
              vo = vfmaq_laneq_f32(vo, vi02, vf00, 2);
              vo = vfmaq_laneq_f32(vo, vi10, vf01, 0);
              vo = vfmaq_laneq_f32(vo, vi11, vf01, 1);
              vo = vfmaq_laneq_f32(vo, vi12, vf01, 2);
              vo = vfmaq_laneq_f32(vo, vi20, vf02, 1);
              vo = vfmaq_laneq_f32(vo, vi21, vf02, 2);
              vo = vfmaq_laneq_f32(vo, vi22, vf02, 3);
#else   // arm v7
              // load filter (1 outch x 3 height x 3 width): vf_outch_height
              float32x2_t vf01 = vld1_bf16(f_ptr);
              float32x2_t vf23 = vld1_bf16(f_ptr + 2);
              float32x2_t vf45 = vld1_bf16(f_ptr + 4);
              float32x2_t vf67 = vld1_bf16(f_ptr + 6);
              float32x2_t vf78 = vld1_bf16(f_ptr + 7);

              // outch 0, height 0
              vo = vmlaq_lane_f32(vo, vi00, vf01, 0);
              vo = vmlaq_lane_f32(vo, vi01, vf01, 1);
              vo = vmlaq_lane_f32(vo, vi02, vf23, 0);
              vo = vmlaq_lane_f32(vo, vi10, vf23, 1);
              vo = vmlaq_lane_f32(vo, vi11, vf45, 0);
              vo = vmlaq_lane_f32(vo, vi12, vf45, 1);
              vo = vmlaq_lane_f32(vo, vi20, vf67, 0);
              vo = vmlaq_lane_f32(vo, vi21, vf67, 1);
              vo = vmlaq_lane_f32(vo, vi22, vf78, 1);
#endif
              in_base += p.in_image_size;
              f_ptr += 9;
            }
            vst1q_bf16(out_base + out_offset, vo);
          }
        }
      }    // m
    }      // b
  }, 0, p.batch, 1, 0, p.out_channels, 1);

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
