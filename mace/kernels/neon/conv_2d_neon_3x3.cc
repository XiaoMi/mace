//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <arm_neon.h>
#include "mace/core/common.h"

namespace mace {
namespace kernels {

static const int REGISTER_SIZE = 4;

void Conv2dNeonK3x3S1(const float* input, // NCHW
                       const index_t* input_shape,
                       const float* filter, // c_out, c_in, kernel_h, kernel_w
                       const float* bias, // c_out
                       float* output, // NCHW
                       const index_t* output_shape) {

  int batch    = output_shape[0];
  int channels = output_shape[1];
  int height   = output_shape[2];
  int width    = output_shape[3];

  int input_batch    = input_shape[0];
  int input_channels = input_shape[1];
  int input_height   = input_shape[2];
  int input_width    = input_shape[3];

  int kernel_h = 3;
  int kernel_w  = 3;

  int height_count = (height >> 1) << 1;
  for (int b = 0; b < batch; ++b) {
    float* output_ptr_base = output + b * channels * height * width;
    for (int oc = 0; oc < channels; ++oc) {
      const float* filter_ptr = filter + oc * input_channels * kernel_h * kernel_w;
      const float* input_ptr = input + b * input_channels * input_height * input_width;
      float* output_ptr = output_ptr_base + oc * height * width;

      std::fill(output_ptr, output_ptr + height * width, bias[oc]);
      for (int ic = 0; ic < input_channels; ++ic) {
        float32x4_t filter0 = vld1q_f32(filter_ptr);
        float32x4_t filter3 = vld1q_f32(filter_ptr+3);
        float32x4_t filter6 = vld1q_f32(filter_ptr+6);

        const float* row[REGISTER_SIZE] = {
                input_ptr, input_ptr + input_width,
                input_ptr + 2 * input_width, input_ptr + 3 * input_width
        };

        float* output_ptr1 = output_ptr;
        float* output_ptr2 = output_ptr + width;

        for (int h = 0; h < height_count; h += 2) {

          int count = width >> 2;
          int remain_count = width & 3;

          for (; count > 0; --count) {
            float32x4_t sum0 = vdupq_n_f32(.0f);
            float32x4_t sum1 = vdupq_n_f32(.0f);
            float32x4_t row0_ext_0 = vld1q_f32(row[0]); //0123
            float32x4_t row0_latter = vld1q_f32(row[0] + REGISTER_SIZE); //4567
            float32x4_t row0_ext_1 = vextq_f32(row0_ext_0, row0_latter, 1); //1234
            float32x4_t row0_ext_2 = vextq_f32(row0_ext_0, row0_latter, 2); //2345

            sum0 = vfmaq_laneq_f32(sum0, row0_ext_0, filter0, 0);
            sum0 = vfmaq_laneq_f32(sum0, row0_ext_1, filter0, 1);
            sum0 = vfmaq_laneq_f32(sum0, row0_ext_2, filter0, 2);

            float32x4_t row1_ext_0 = vld1q_f32(row[1]); //0123
            float32x4_t row1_latter = vld1q_f32(row[1] + REGISTER_SIZE); //4567
            float32x4_t row1_ext_1 = vextq_f32(row1_ext_0, row1_latter, 1); //1234
            float32x4_t row1_ext_2 = vextq_f32(row1_ext_0, row1_latter, 2); //2345

            sum0 = vfmaq_laneq_f32(sum0, row1_ext_0, filter3, 0);
            sum0 = vfmaq_laneq_f32(sum0, row1_ext_1, filter3, 1);
            sum0 = vfmaq_laneq_f32(sum0, row1_ext_2, filter3, 2);

            row0_ext_0 = vld1q_f32(row[2]); //0123
            row0_latter = vld1q_f32(row[2] + REGISTER_SIZE); //4567
            row0_ext_1 = vextq_f32(row0_ext_0, row0_latter, 1); //1234
            row0_ext_2 = vextq_f32(row0_ext_0, row0_latter, 2); //2345

            sum0 = vfmaq_laneq_f32(sum0, row0_ext_0, filter6, 0);
            sum0 = vfmaq_laneq_f32(sum0, row0_ext_1, filter6, 1);
            sum0 = vfmaq_laneq_f32(sum0, row0_ext_2, filter6, 2);

            // second row
            sum1 = vfmaq_laneq_f32(sum1, row1_ext_0, filter0, 0);
            sum1 = vfmaq_laneq_f32(sum1, row1_ext_1, filter0, 1);
            sum1 = vfmaq_laneq_f32(sum1, row1_ext_2, filter0, 2);

            sum1 = vfmaq_laneq_f32(sum1, row0_ext_0, filter3, 0);
            sum1 = vfmaq_laneq_f32(sum1, row0_ext_1, filter3, 1);
            sum1 = vfmaq_laneq_f32(sum1, row0_ext_2, filter3, 2);

            row1_ext_0 = vld1q_f32(row[3]); //0123
            row1_latter = vld1q_f32(row[3] + REGISTER_SIZE); //4567
            row1_ext_1 = vextq_f32(row1_ext_0, row1_latter, 1); //1234
            row1_ext_2 = vextq_f32(row1_ext_0, row1_latter, 2); //2345

            sum1 = vfmaq_laneq_f32(sum1, row1_ext_0, filter6, 0);
            sum1 = vfmaq_laneq_f32(sum1, row1_ext_1, filter6, 1);
            sum1 = vfmaq_laneq_f32(sum1, row1_ext_2, filter6, 2);

            float32x4_t output_row0 = vld1q_f32(output_ptr1);
            float32x4_t output_row1 = vld1q_f32(output_ptr2);
            output_row0 = vaddq_f32(output_row0, sum0);
            output_row1 = vaddq_f32(output_row1, sum1);
            vst1q_f32(output_ptr1, output_row0);
            vst1q_f32(output_ptr2, output_row1);

            output_ptr1 += REGISTER_SIZE;
            output_ptr2 += REGISTER_SIZE;
            for(int i = 0; i < REGISTER_SIZE; ++i) {
              row[i] += REGISTER_SIZE;
            }
          }
          for (; remain_count > 0; --remain_count) {
            float32x4_t row0 = vld1q_f32(row[0]); //0123
            float32x4_t row1 = vld1q_f32(row[1]); //0123
            float32x4_t row2 = vld1q_f32(row[2]); //0123
            float32x4_t row3 = vld1q_f32(row[3]); //0123

            float32x4_t sum = vmulq_f32(row0, filter0);
            sum = vmlaq_f32(sum, row1, filter3);
            sum = vmlaq_f32(sum, row2, filter6);
            sum = vsetq_lane_f32(*output_ptr1, sum, 3);
            *output_ptr1 = vaddvq_f32(sum);

            sum = vmulq_f32(row1, filter0);
            sum = vmlaq_f32(sum, row2, filter3);
            sum = vmlaq_f32(sum, row3, filter6);
            sum = vsetq_lane_f32(*output_ptr2, sum, 3);
            *output_ptr2 = vaddvq_f32(sum);

            ++output_ptr1;
            ++output_ptr2;
            for(int i = 0; i < REGISTER_SIZE; ++i) {
              row[i] += 1;
            }
          }
          output_ptr1 += width;
          output_ptr2 += width;
          for(int i = 0; i < REGISTER_SIZE; ++i) {
            row[i] += 2 + input_width;
          }
        }

        if (height != height_count) {
          int count = width >> 2;
          int remain_count = width & 3;
          for(; count > 0; --count) {
            float32x4_t sum0 = vdupq_n_f32(.0f);
            float32x4_t row0_ext_0 = vld1q_f32(row[0]); //0123
            float32x4_t row0_latter = vld1q_f32(row[0] + REGISTER_SIZE); //4567
            float32x4_t row0_ext_1 = vextq_f32(row0_ext_0, row0_latter, 1); //1234
            float32x4_t row0_ext_2 = vextq_f32(row0_ext_0, row0_latter, 2); //2345

            sum0 = vfmaq_laneq_f32(sum0, row0_ext_0, filter0, 0);
            sum0 = vfmaq_laneq_f32(sum0, row0_ext_1, filter0, 1);
            sum0 = vfmaq_laneq_f32(sum0, row0_ext_2, filter0, 2);

            float32x4_t row1_ext_0 = vld1q_f32(row[1]); //0123
            float32x4_t row1_latter = vld1q_f32(row[1] + REGISTER_SIZE); //4567
            float32x4_t row1_ext_1 = vextq_f32(row1_ext_0, row1_latter, 1); //1234
            float32x4_t row1_ext_2 = vextq_f32(row1_ext_0, row1_latter, 2); //2345

            sum0 = vfmaq_laneq_f32(sum0, row1_ext_0, filter3, 0);
            sum0 = vfmaq_laneq_f32(sum0, row1_ext_1, filter3, 1);
            sum0 = vfmaq_laneq_f32(sum0, row1_ext_2, filter3, 2);

            row0_ext_0 = vld1q_f32(row[2]); //0123
            row0_latter = vld1q_f32(row[2] + REGISTER_SIZE); //4567
            row0_ext_1 = vextq_f32(row0_ext_0, row0_latter, 1); //1234
            row0_ext_2 = vextq_f32(row0_ext_0, row0_latter, 2); //2345

            sum0 = vfmaq_laneq_f32(sum0, row0_ext_0, filter6, 0);
            sum0 = vfmaq_laneq_f32(sum0, row0_ext_1, filter6, 1);
            sum0 = vfmaq_laneq_f32(sum0, row0_ext_2, filter6, 2);

            float32x4_t output_row0 = vld1q_f32(output_ptr1);
            output_row0 = vaddq_f32(output_row0, sum0);
            vst1q_f32(output_ptr1, output_row0);
            output_ptr1 += REGISTER_SIZE;
            for(int i = 0; i < 3; ++i) {
              row[i] += REGISTER_SIZE;
            }
          }
          for (; remain_count > 0; --remain_count) {
            float32x4_t row0 = vld1q_f32(row[0]); //0123
            float32x4_t row1 = vld1q_f32(row[1]); //0123
            float32x4_t row2 = vld1q_f32(row[2]); //0123

            float32x4_t sum = vmulq_f32(row0, filter0);
            sum = vmlaq_f32(sum, row1, filter3);
            sum = vmlaq_f32(sum, row2, filter6);
            sum = vsetq_lane_f32(*output_ptr1, sum, 3);
            *output_ptr1 = vaddvq_f32(sum);

            ++output_ptr1;
            for(int i = 0; i < 3; ++i) {
              row[i] += 1;
            }
          }
        }
        filter_ptr += 9;
        input_ptr += input_height * input_width;
      }
    }
  }
}

} //  namespace kernels
} //  namespace mace
