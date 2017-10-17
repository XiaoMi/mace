//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#ifndef MACE_KERNELS_NEON_CONV_2D_NEON_5X5_H_
#define MACE_KERNELS_NEON_CONV_2D_NEON_5X5_H_

#include <arm_neon.h>
#include "mace/core/common.h"

namespace mace {
namespace kernels {

void Conv2dNeonK5x5S1(const float *input,  // NCHW
                      const index_t *input_shape,
                      const float *filter,  // c_out, c_in, kernel_h, kernel_w
                      const index_t *filter_shape,
                      const float *bias,  // c_out
                      float *output,      // NCHW
                      const index_t *output_shape) {
  const index_t batch = output_shape[0];
  const index_t channels = output_shape[1];
  const index_t height = output_shape[2];
  const index_t width = output_shape[3];

  const index_t input_batch = input_shape[0];
  const index_t input_channels = input_shape[1];
  const index_t input_height = input_shape[2];
  const index_t input_width = input_shape[3];

  MACE_ASSERT(input_batch == batch);

  const index_t input_total_pixels_per_channel = input_height * input_width;
  const index_t output_total_pixels_per_channel = height * width;
  const index_t input_total_pixels_per_batch =
      input_total_pixels_per_channel * input_channels;
  const index_t output_total_pixels_per_batch =
      output_total_pixels_per_channel * channels;
  const index_t patch_size = input_channels * 25;

#pragma omp parallel for collapse(2)
  for (index_t n = 0; n < batch; ++n) {
    for (index_t c = 0; c < channels; ++c) {
      float *output_ptr = output + n * output_total_pixels_per_batch +
                          c * output_total_pixels_per_channel;
      const float *input_ptr = input + n * input_total_pixels_per_batch;

      // Fill with bias
      std::fill(output_ptr, output_ptr + output_total_pixels_per_channel,
                bias ? bias[c] : 0);

      for (index_t inc = 0; inc < input_channels; ++inc) {
        float *outptr = output_ptr;
        float *outptr2 = outptr + width;

        const float *inptr = input_ptr + inc * input_total_pixels_per_channel;
        const float *filter_ptr = filter + c * patch_size + inc * 25;

        const float *r0 = inptr;
        const float *r1 = inptr + input_width;
        const float *r2 = inptr + input_width * 2;
        const float *r3 = inptr + input_width * 3;
        const float *r4 = inptr + input_width * 4;
        const float *r5 = inptr + input_width * 5;

        const float *k0 = filter_ptr;
        const float *k1 = filter_ptr + 5;
        const float *k2 = filter_ptr + 10;
        const float *k3 = filter_ptr + 15;
        const float *k4 = filter_ptr + 20;

        float32x4_t _k0123 = vld1q_f32(filter_ptr);
        float32x4_t _k4567 = vld1q_f32(filter_ptr + 4);
        float32x4_t _k891011 = vld1q_f32(filter_ptr + 8);
        float32x4_t _k12131415 = vld1q_f32(filter_ptr + 12);
        float32x4_t _k16171819 = vld1q_f32(filter_ptr + 16);
        float32x4_t _k20212223 = vld1q_f32(filter_ptr + 20);
        float32x4_t _k24242424 = vdupq_n_f32(filter_ptr[24]);

        // height_block_size = 2, width_block_size = 4
        int h = 0;
        for (; h + 1 < height; h += 2) {
          int width_blocks = width >> 2;
          int remain = width - (width_blocks << 2);

          for (; width_blocks > 0; --width_blocks) {
            float32x4_t _sum = vld1q_f32(outptr);
            float32x4_t _sum2 = vld1q_f32(outptr2);

            float32x4_t _r00 = vld1q_f32(r0);
            float32x4_t _r04 = vld1q_f32(r0 + 4);
            float32x4_t _r01 = vextq_f32(_r00, _r04, 1);
            float32x4_t _r02 = vextq_f32(_r00, _r04, 2);
            float32x4_t _r03 = vextq_f32(_r00, _r04, 3);

            float32x4_t _r10 = vld1q_f32(r1);
            float32x4_t _r14 = vld1q_f32(r1 + 4);
            float32x4_t _r11 = vextq_f32(_r10, _r14, 1);
            float32x4_t _r12 = vextq_f32(_r10, _r14, 2);
            float32x4_t _r13 = vextq_f32(_r10, _r14, 3);

            float32x4_t _r20 = vld1q_f32(r2);
            float32x4_t _r24 = vld1q_f32(r2 + 4);
            float32x4_t _r21 = vextq_f32(_r20, _r24, 1);
            float32x4_t _r22 = vextq_f32(_r20, _r24, 2);
            float32x4_t _r23 = vextq_f32(_r20, _r24, 3);

            float32x4_t _r30 = vld1q_f32(r3);
            float32x4_t _r34 = vld1q_f32(r3 + 4);
            float32x4_t _r31 = vextq_f32(_r30, _r34, 1);
            float32x4_t _r32 = vextq_f32(_r30, _r34, 2);
            float32x4_t _r33 = vextq_f32(_r30, _r34, 3);

            float32x4_t _r40 = vld1q_f32(r4);
            float32x4_t _r44 = vld1q_f32(r4 + 4);
            float32x4_t _r41 = vextq_f32(_r40, _r44, 1);
            float32x4_t _r42 = vextq_f32(_r40, _r44, 2);
            float32x4_t _r43 = vextq_f32(_r40, _r44, 3);

            float32x4_t _r50 = vld1q_f32(r5);
            float32x4_t _r54 = vld1q_f32(r5 + 4);
            float32x4_t _r51 = vextq_f32(_r50, _r54, 1);
            float32x4_t _r52 = vextq_f32(_r50, _r54, 2);
            float32x4_t _r53 = vextq_f32(_r50, _r54, 3);

            _sum = vfmaq_laneq_f32(_sum, _r00, _k0123, 0);
            _sum = vfmaq_laneq_f32(_sum, _r01, _k0123, 1);
            _sum = vfmaq_laneq_f32(_sum, _r02, _k0123, 2);
            _sum = vfmaq_laneq_f32(_sum, _r03, _k0123, 3);
            _sum = vfmaq_laneq_f32(_sum, _r04, _k4567, 0);

            _sum = vfmaq_laneq_f32(_sum, _r10, _k4567, 1);
            _sum = vfmaq_laneq_f32(_sum, _r11, _k4567, 2);
            _sum = vfmaq_laneq_f32(_sum, _r12, _k4567, 3);
            _sum = vfmaq_laneq_f32(_sum, _r13, _k891011, 0);
            _sum = vfmaq_laneq_f32(_sum, _r14, _k891011, 1);

            _sum = vfmaq_laneq_f32(_sum, _r20, _k891011, 2);
            _sum = vfmaq_laneq_f32(_sum, _r21, _k891011, 3);
            _sum = vfmaq_laneq_f32(_sum, _r22, _k12131415, 0);
            _sum = vfmaq_laneq_f32(_sum, _r23, _k12131415, 1);
            _sum = vfmaq_laneq_f32(_sum, _r24, _k12131415, 2);

            _sum = vfmaq_laneq_f32(_sum, _r30, _k12131415, 3);
            _sum = vfmaq_laneq_f32(_sum, _r31, _k16171819, 0);
            _sum = vfmaq_laneq_f32(_sum, _r32, _k16171819, 1);
            _sum = vfmaq_laneq_f32(_sum, _r33, _k16171819, 2);
            _sum = vfmaq_laneq_f32(_sum, _r34, _k16171819, 3);

            _sum = vfmaq_laneq_f32(_sum, _r40, _k20212223, 0);
            _sum = vfmaq_laneq_f32(_sum, _r41, _k20212223, 1);
            _sum = vfmaq_laneq_f32(_sum, _r42, _k20212223, 2);
            _sum = vfmaq_laneq_f32(_sum, _r43, _k20212223, 3);
            _sum = vfmaq_laneq_f32(_sum, _r44, _k24242424, 0);

            _sum2 = vfmaq_laneq_f32(_sum2, _r10, _k0123, 0);
            _sum2 = vfmaq_laneq_f32(_sum2, _r11, _k0123, 1);
            _sum2 = vfmaq_laneq_f32(_sum2, _r12, _k0123, 2);
            _sum2 = vfmaq_laneq_f32(_sum2, _r13, _k0123, 3);
            _sum2 = vfmaq_laneq_f32(_sum2, _r14, _k4567, 0);

            _sum2 = vfmaq_laneq_f32(_sum2, _r20, _k4567, 1);
            _sum2 = vfmaq_laneq_f32(_sum2, _r21, _k4567, 2);
            _sum2 = vfmaq_laneq_f32(_sum2, _r22, _k4567, 3);
            _sum2 = vfmaq_laneq_f32(_sum2, _r23, _k891011, 0);
            _sum2 = vfmaq_laneq_f32(_sum2, _r24, _k891011, 1);

            _sum2 = vfmaq_laneq_f32(_sum2, _r30, _k891011, 2);
            _sum2 = vfmaq_laneq_f32(_sum2, _r31, _k891011, 3);
            _sum2 = vfmaq_laneq_f32(_sum2, _r32, _k12131415, 0);
            _sum2 = vfmaq_laneq_f32(_sum2, _r33, _k12131415, 1);
            _sum2 = vfmaq_laneq_f32(_sum2, _r34, _k12131415, 2);

            _sum2 = vfmaq_laneq_f32(_sum2, _r40, _k12131415, 3);
            _sum2 = vfmaq_laneq_f32(_sum2, _r41, _k16171819, 0);
            _sum2 = vfmaq_laneq_f32(_sum2, _r42, _k16171819, 1);
            _sum2 = vfmaq_laneq_f32(_sum2, _r43, _k16171819, 2);
            _sum2 = vfmaq_laneq_f32(_sum2, _r44, _k16171819, 3);

            _sum2 = vfmaq_laneq_f32(_sum2, _r50, _k20212223, 0);
            _sum2 = vfmaq_laneq_f32(_sum2, _r51, _k20212223, 1);
            _sum2 = vfmaq_laneq_f32(_sum2, _r52, _k20212223, 2);
            _sum2 = vfmaq_laneq_f32(_sum2, _r53, _k20212223, 3);
            _sum2 = vfmaq_laneq_f32(_sum2, _r54, _k24242424, 0);

            vst1q_f32(outptr, _sum);
            vst1q_f32(outptr2, _sum2);

            r0 += 4;
            r1 += 4;
            r2 += 4;
            r3 += 4;
            r4 += 4;
            r5 += 4;
            outptr += 4;
            outptr2 += 4;
          }

          for (; remain > 0; --remain) {
            float sum = 0;
            float sum2 = 0;

            float32x4_t _r1 = vld1q_f32(r1);
            float32x4_t _k1 = vld1q_f32(k1);
            float32x4_t _sum = vmulq_f32(_r1, _k1);
            float32x4_t _sum2 = vmulq_f32(_r1, _k0123);

            float32x4_t _r2 = vld1q_f32(r2);
            float32x4_t _k2 = vld1q_f32(k2);
            _sum = vmlaq_f32(_sum, _r2, _k2);
            _sum2 = vmlaq_f32(_sum2, _r2, _k1);

            float32x4_t _r3 = vld1q_f32(r3);
            float32x4_t _k3 = vld1q_f32(k3);
            _sum = vmlaq_f32(_sum, _r3, _k3);
            _sum2 = vmlaq_f32(_sum2, _r3, _k2);

            float32x4_t _r4 = vld1q_f32(r4);
            _sum = vmlaq_f32(_sum, _r4, _k20212223);
            _sum2 = vmlaq_f32(_sum2, _r4, _k3);

            float32x4_t _r0 = vld1q_f32(r0);
            _sum = vmlaq_f32(_sum, _r0, _k0123);
            float32x4_t _r5 = vld1q_f32(r5);
            _sum2 = vmlaq_f32(_sum2, _r5, _k20212223);

            float32x4_t _k_t4;
            _k_t4 = vsetq_lane_f32(k0[4], _k_t4, 0);
            _k_t4 = vsetq_lane_f32(k1[4], _k_t4, 1);
            _k_t4 = vsetq_lane_f32(k2[4], _k_t4, 2);
            _k_t4 = vsetq_lane_f32(k3[4], _k_t4, 3);

            float32x4_t _r_t4;

            _r_t4 = vsetq_lane_f32(r0[4], _r_t4, 0);
            _r_t4 = vsetq_lane_f32(r1[4], _r_t4, 1);
            _r_t4 = vsetq_lane_f32(r2[4], _r_t4, 2);
            _r_t4 = vsetq_lane_f32(r3[4], _r_t4, 3);
            _sum = vmlaq_f32(_sum, _r_t4, _k_t4);

            sum = r4[4] * k4[4];

            _r_t4 = vextq_f32(_r_t4, _r_t4, 1);
            _r_t4 = vsetq_lane_f32(r4[4], _r_t4, 3);
            _sum2 = vmlaq_f32(_sum2, _r_t4, _k_t4);

            sum2 = r5[4] * k4[4];

            float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
            float32x2_t _ss2 =
                vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
            float32x2_t _ss_ss2 = vpadd_f32(_ss, _ss2);

            sum += vget_lane_f32(_ss_ss2, 0);
            sum2 += vget_lane_f32(_ss_ss2, 1);

            *outptr += sum;
            *outptr2 += sum2;

            ++r0;
            ++r1;
            ++r2;
            ++r3;
            ++r4;
            ++r5;
            ++outptr;
            ++outptr2;
          }

          r0 += 4 + input_width;  // 4 = 5 - 1
          r1 += 4 + input_width;
          r2 += 4 + input_width;
          r3 += 4 + input_width;
          r4 += 4 + input_width;
          r5 += 4 + input_width;
          outptr += width;
          outptr2 += width;
        }

        for (; h < height; ++h) {
          // may left one row if odd rows
          int width_blocks = width >> 2;
          int remain = width - (width_blocks << 2);
          for (; width_blocks > 0; --width_blocks) {
            float32x4_t _sum = vld1q_f32(outptr);

            float32x4_t _r00 = vld1q_f32(r0);
            float32x4_t _r04 = vld1q_f32(r0 + 4);
            float32x4_t _r01 = vextq_f32(_r00, _r04, 1);
            float32x4_t _r02 = vextq_f32(_r00, _r04, 2);
            float32x4_t _r03 = vextq_f32(_r00, _r04, 3);

            float32x4_t _r10 = vld1q_f32(r1);
            float32x4_t _r14 = vld1q_f32(r1 + 4);
            float32x4_t _r11 = vextq_f32(_r10, _r14, 1);
            float32x4_t _r12 = vextq_f32(_r10, _r14, 2);
            float32x4_t _r13 = vextq_f32(_r10, _r14, 3);

            float32x4_t _r20 = vld1q_f32(r2);
            float32x4_t _r24 = vld1q_f32(r2 + 4);
            float32x4_t _r21 = vextq_f32(_r20, _r24, 1);
            float32x4_t _r22 = vextq_f32(_r20, _r24, 2);
            float32x4_t _r23 = vextq_f32(_r20, _r24, 3);

            float32x4_t _r30 = vld1q_f32(r3);
            float32x4_t _r34 = vld1q_f32(r3 + 4);
            float32x4_t _r31 = vextq_f32(_r30, _r34, 1);
            float32x4_t _r32 = vextq_f32(_r30, _r34, 2);
            float32x4_t _r33 = vextq_f32(_r30, _r34, 3);

            float32x4_t _r40 = vld1q_f32(r4);
            float32x4_t _r44 = vld1q_f32(r4 + 4);
            float32x4_t _r41 = vextq_f32(_r40, _r44, 1);
            float32x4_t _r42 = vextq_f32(_r40, _r44, 2);
            float32x4_t _r43 = vextq_f32(_r40, _r44, 3);

            _sum = vfmaq_laneq_f32(_sum, _r00, _k0123, 0);
            _sum = vfmaq_laneq_f32(_sum, _r01, _k0123, 1);
            _sum = vfmaq_laneq_f32(_sum, _r02, _k0123, 2);
            _sum = vfmaq_laneq_f32(_sum, _r03, _k0123, 3);
            _sum = vfmaq_laneq_f32(_sum, _r04, _k4567, 0);

            _sum = vfmaq_laneq_f32(_sum, _r10, _k4567, 1);
            _sum = vfmaq_laneq_f32(_sum, _r11, _k4567, 2);
            _sum = vfmaq_laneq_f32(_sum, _r12, _k4567, 3);
            _sum = vfmaq_laneq_f32(_sum, _r13, _k891011, 0);
            _sum = vfmaq_laneq_f32(_sum, _r14, _k891011, 1);

            _sum = vfmaq_laneq_f32(_sum, _r20, _k891011, 2);
            _sum = vfmaq_laneq_f32(_sum, _r21, _k891011, 3);
            _sum = vfmaq_laneq_f32(_sum, _r22, _k12131415, 0);
            _sum = vfmaq_laneq_f32(_sum, _r23, _k12131415, 1);
            _sum = vfmaq_laneq_f32(_sum, _r24, _k12131415, 2);

            _sum = vfmaq_laneq_f32(_sum, _r30, _k12131415, 3);
            _sum = vfmaq_laneq_f32(_sum, _r31, _k16171819, 0);
            _sum = vfmaq_laneq_f32(_sum, _r32, _k16171819, 1);
            _sum = vfmaq_laneq_f32(_sum, _r33, _k16171819, 2);
            _sum = vfmaq_laneq_f32(_sum, _r34, _k16171819, 3);

            _sum = vfmaq_laneq_f32(_sum, _r40, _k20212223, 0);
            _sum = vfmaq_laneq_f32(_sum, _r41, _k20212223, 1);
            _sum = vfmaq_laneq_f32(_sum, _r42, _k20212223, 2);
            _sum = vfmaq_laneq_f32(_sum, _r43, _k20212223, 3);
            _sum = vfmaq_laneq_f32(_sum, _r44, _k24242424, 0);

            vst1q_f32(outptr, _sum);

            r0 += 4;
            r1 += 4;
            r2 += 4;
            r3 += 4;
            r4 += 4;
            r5 += 4;
            outptr += 4;
          }

          for (; remain > 0; --remain) {
            float sum = 0;
            float32x4_t _r0 = vld1q_f32(r0);
            float32x4_t _sum = vmulq_f32(_r0, _k0123);

            float debug[4];
            vst1q_f32(debug, _sum);

            float32x4_t _r1 = vld1q_f32(r1);
            _sum = vmlaq_f32(_sum, _r1, vld1q_f32(k1));

            float32x4_t _r2 = vld1q_f32(r2);
            _sum = vmlaq_f32(_sum, _r2, vld1q_f32(k2));

            float32x4_t _r3 = vld1q_f32(r3);
            _sum = vmlaq_f32(_sum, _r3, vld1q_f32(k3));

            float32x4_t _r4 = vld1q_f32(r4);
            _sum = vmlaq_f32(_sum, _r4, _k20212223);

            float32x4_t _k_t4;
            _k_t4 = vsetq_lane_f32(k0[4], _k_t4, 0);
            _k_t4 = vsetq_lane_f32(k1[4], _k_t4, 1);
            _k_t4 = vsetq_lane_f32(k2[4], _k_t4, 2);
            _k_t4 = vsetq_lane_f32(k3[4], _k_t4, 3);

            float32x4_t _r_t4;

            _r_t4 = vsetq_lane_f32(r0[4], _r_t4, 0);
            _r_t4 = vsetq_lane_f32(r1[4], _r_t4, 1);
            _r_t4 = vsetq_lane_f32(r2[4], _r_t4, 2);
            _r_t4 = vsetq_lane_f32(r3[4], _r_t4, 3);
            _sum = vmlaq_f32(_sum, _r_t4, _k_t4);

            sum = r4[4] * k4[4];

            float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
            _ss = vpadd_f32(_ss, _ss);

            sum += vget_lane_f32(_ss, 0);
            *outptr += sum;

            ++r0;
            ++r1;
            ++r2;
            ++r3;
            ++r4;
            ++outptr;
          }
          r0 += 4;
          r1 += 4;
          r2 += 4;
          r3 += 4;
          r4 += 4;
        }
      }
    }
  }
}

}  //  namespace kernels
}  //  namespace mace

#endif  //  MACE_KERNELS_NEON_CONV_2D_NEON_5X5_H_
