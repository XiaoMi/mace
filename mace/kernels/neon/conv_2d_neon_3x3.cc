//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <arm_neon.h>

namespace mace {
namespace kernels {

static const int kRegisterSize = 4;
static const int kFilterSize = 9;

void Conv2dNeonK3x3S1(const float *input,  // NCHW
                      const index_t *input_shape,
                      const float *filter,  // c_out, c_in, kernel_h, kernel_w
                      const index_t *filter_shape,
                      const float *bias,  // c_out
                      float *output,      // NCHW
                      const index_t *output_shape) {
  int height_count = (output_shape[2] >> 1) << 1;

  int output_batch = output_shape[0];
  int output_channels = output_shape[1];
  int output_height = output_shape[2];
  int output_width = output_shape[3];
  int input_batch = input_shape[0];
  int input_channels = input_shape[1];
  int input_height = input_shape[2];
  int input_width = input_shape[3];
  int multiplier =
      filter_shape == nullptr ? 0 : filter_shape[0];
  int filter_in_channels =
      filter_shape == nullptr ? input_channels : 1;
#pragma omp parallel for collapse(2)
  for (int b = 0; b < output_batch; ++b) {
    for (int oc = 0; oc < output_channels; ++oc) {
      float *output_ptr_base =
          output + b * output_channels * output_height * output_width;
      const float *filter_ptr = filter + oc * filter_in_channels * kFilterSize;
      const float *input_ptr =
          input + b * input_channels * input_height * input_width;
      if (filter_shape != nullptr) {
        input_ptr += (oc / multiplier) * input_height * input_width;
      }
      float *output_ptr = output_ptr_base + oc * output_height * output_width;
      std::fill(output_ptr, output_ptr + output_height * output_width,
                bias ? bias[oc] : 0);
      for (int ic = 0; ic < filter_in_channels; ++ic) {
        float32x4_t n_filter_v[3] = {vld1q_f32(filter_ptr),
                                     vld1q_f32(filter_ptr + 3),
                                     vld1q_f32(filter_ptr + 6)};

        const float *row_ptr_v[kRegisterSize] = {
            input_ptr, input_ptr + input_width, input_ptr + 2 * input_width,
            input_ptr + 3 * input_width};

        float *output_ptr_v[] = {output_ptr, output_ptr + output_width};

        for (int h = 0; h < height_count; h += 2) {
          int count = output_width >> 2;
          int remain_count = output_width & 3;

          for (; count > 0; --count) {
            float32x4_t n_sum0 = vdupq_n_f32(.0f);

            float32x4_t n_row_former = vld1q_f32(row_ptr_v[0]);
            float32x4_t n_row_latter = vld1q_f32(row_ptr_v[0] + kRegisterSize);
            float32x4_t n_row_ext0 = vextq_f32(n_row_former, n_row_latter, 1);
            float32x4_t n_row_ext1 = vextq_f32(n_row_former, n_row_latter, 2);
            n_sum0 = vfmaq_laneq_f32(n_sum0, n_row_former, n_filter_v[0], 0);
            n_sum0 = vfmaq_laneq_f32(n_sum0, n_row_ext0, n_filter_v[0], 1);
            n_sum0 = vfmaq_laneq_f32(n_sum0, n_row_ext1, n_filter_v[0], 2);

            float32x4_t n_row1_former = vld1q_f32(row_ptr_v[1]);
            float32x4_t n_row1_latter = vld1q_f32(row_ptr_v[1] + kRegisterSize);
            float32x4_t n_row1_ext0 =
                vextq_f32(n_row1_former, n_row1_latter, 1);
            float32x4_t n_row1_ext1 =
                vextq_f32(n_row1_former, n_row1_latter, 2);
            n_sum0 = vfmaq_laneq_f32(n_sum0, n_row1_former, n_filter_v[1], 0);
            n_sum0 = vfmaq_laneq_f32(n_sum0, n_row1_ext0, n_filter_v[1], 1);
            n_sum0 = vfmaq_laneq_f32(n_sum0, n_row1_ext1, n_filter_v[1], 2);

            n_row_former = vld1q_f32(row_ptr_v[2]);
            n_row_latter = vld1q_f32(row_ptr_v[2] + kRegisterSize);
            n_row_ext0 = vextq_f32(n_row_former, n_row_latter, 1);
            n_row_ext1 = vextq_f32(n_row_former, n_row_latter, 2);
            n_sum0 = vfmaq_laneq_f32(n_sum0, n_row_former, n_filter_v[2], 0);
            n_sum0 = vfmaq_laneq_f32(n_sum0, n_row_ext0, n_filter_v[2], 1);
            n_sum0 = vfmaq_laneq_f32(n_sum0, n_row_ext1, n_filter_v[2], 2);

            // second row
            float32x4_t n_sum1 = vdupq_n_f32(.0f);

            n_sum1 = vfmaq_laneq_f32(n_sum1, n_row1_former, n_filter_v[0], 0);
            n_sum1 = vfmaq_laneq_f32(n_sum1, n_row1_ext0, n_filter_v[0], 1);
            n_sum1 = vfmaq_laneq_f32(n_sum1, n_row1_ext1, n_filter_v[0], 2);

            n_sum1 = vfmaq_laneq_f32(n_sum1, n_row_former, n_filter_v[1], 0);
            n_sum1 = vfmaq_laneq_f32(n_sum1, n_row_ext0, n_filter_v[1], 1);
            n_sum1 = vfmaq_laneq_f32(n_sum1, n_row_ext1, n_filter_v[1], 2);

            n_row1_former = vld1q_f32(row_ptr_v[3]);
            n_row1_latter = vld1q_f32(row_ptr_v[3] + kRegisterSize);
            n_row1_ext0 = vextq_f32(n_row1_former, n_row1_latter, 1);
            n_row1_ext1 = vextq_f32(n_row1_former, n_row1_latter, 2);
            n_sum1 = vfmaq_laneq_f32(n_sum1, n_row1_former, n_filter_v[2], 0);
            n_sum1 = vfmaq_laneq_f32(n_sum1, n_row1_ext0, n_filter_v[2], 1);
            n_sum1 = vfmaq_laneq_f32(n_sum1, n_row1_ext1, n_filter_v[2], 2);

            float32x4_t n_output_row = vld1q_f32(output_ptr_v[0]);
            float32x4_t n_output_row1 = vld1q_f32(output_ptr_v[1]);
            n_output_row = vaddq_f32(n_output_row, n_sum0);
            n_output_row1 = vaddq_f32(n_output_row1, n_sum1);
            vst1q_f32(output_ptr_v[0], n_output_row);
            vst1q_f32(output_ptr_v[1], n_output_row1);
            output_ptr_v[0] += kRegisterSize;
            output_ptr_v[1] += kRegisterSize;
            for (int i = 0; i < kRegisterSize; ++i) {
              row_ptr_v[i] += kRegisterSize;
            }
          }
          for (; remain_count > 0; --remain_count) {
            float32x4_t n_row_v[] = {vld1q_f32(row_ptr_v[0]),
                                     vld1q_f32(row_ptr_v[1]),
                                     vld1q_f32(row_ptr_v[2])};
            float32x4_t n_sum0 = vmulq_f32(n_row_v[0], n_filter_v[0]);
            n_sum0 = vmlaq_f32(n_sum0, n_row_v[1], n_filter_v[1]);
            n_sum0 = vmlaq_f32(n_sum0, n_row_v[2], n_filter_v[2]);
            n_sum0 = vsetq_lane_f32(*output_ptr_v[0], n_sum0, 3);
            *output_ptr_v[0] = vaddvq_f32(n_sum0);

            float32x4_t n_row3 = vld1q_f32(row_ptr_v[3]);
            float32x4_t n_sum1 = vmulq_f32(n_row_v[1], n_filter_v[0]);
            n_sum1 = vmlaq_f32(n_sum1, n_row_v[2], n_filter_v[1]);
            n_sum1 = vmlaq_f32(n_sum1, n_row3, n_filter_v[2]);
            n_sum1 = vsetq_lane_f32(*output_ptr_v[1], n_sum1, 3);
            *output_ptr_v[1] = vaddvq_f32(n_sum1);

            ++output_ptr_v[0];
            ++output_ptr_v[1];
            for (int i = 0; i < kRegisterSize; ++i) {
              row_ptr_v[i] += 1;
            }
          }
          output_ptr_v[0] += output_width;
          output_ptr_v[1] += output_width;
          for (int i = 0; i < kRegisterSize; ++i) {
            row_ptr_v[i] += 2 + input_width;
          }
        }

        if (output_height != height_count) {
          int count = output_width >> 2;
          int remain_count = output_width & 3;
          for (; count > 0; --count) {
            float32x4_t n_sum = vdupq_n_f32(.0f);
            float32x4_t n_row_former = vld1q_f32(row_ptr_v[0]);
            float32x4_t n_row_latter = vld1q_f32(row_ptr_v[0] + kRegisterSize);
            float32x4_t n_row_ext1 = vextq_f32(n_row_former, n_row_latter, 1);
            float32x4_t n_row_ext2 = vextq_f32(n_row_former, n_row_latter, 2);
            n_sum = vfmaq_laneq_f32(n_sum, n_row_former, n_filter_v[0], 0);
            n_sum = vfmaq_laneq_f32(n_sum, n_row_ext1, n_filter_v[0], 1);
            n_sum = vfmaq_laneq_f32(n_sum, n_row_ext2, n_filter_v[0], 2);

            n_row_former = vld1q_f32(row_ptr_v[1]);
            n_row_latter = vld1q_f32(row_ptr_v[1] + kRegisterSize);
            n_row_ext1 = vextq_f32(n_row_former, n_row_latter, 1);
            n_row_ext2 = vextq_f32(n_row_former, n_row_latter, 2);
            n_sum = vfmaq_laneq_f32(n_sum, n_row_former, n_filter_v[1], 0);
            n_sum = vfmaq_laneq_f32(n_sum, n_row_ext1, n_filter_v[1], 1);
            n_sum = vfmaq_laneq_f32(n_sum, n_row_ext2, n_filter_v[1], 2);

            n_row_former = vld1q_f32(row_ptr_v[2]);
            n_row_latter = vld1q_f32(row_ptr_v[2] + kRegisterSize);
            n_row_ext1 = vextq_f32(n_row_former, n_row_latter, 1);
            n_row_ext2 = vextq_f32(n_row_former, n_row_latter, 2);
            n_sum = vfmaq_laneq_f32(n_sum, n_row_former, n_filter_v[2], 0);
            n_sum = vfmaq_laneq_f32(n_sum, n_row_ext1, n_filter_v[2], 1);
            n_sum = vfmaq_laneq_f32(n_sum, n_row_ext2, n_filter_v[2], 2);

            float32x4_t n_output_row = vld1q_f32(output_ptr_v[0]);
            n_output_row = vaddq_f32(n_output_row, n_sum);
            vst1q_f32(output_ptr_v[0], n_output_row);
            output_ptr_v[0] += kRegisterSize;
            for (int i = 0; i < 3; ++i) {
              row_ptr_v[i] += kRegisterSize;
            }
          }
          for (; remain_count > 0; --remain_count) {
            float32x4_t n_row_v[] = {
                vld1q_f32(row_ptr_v[0]), vld1q_f32(row_ptr_v[1]),
                vld1q_f32(row_ptr_v[2]),
            };

            float32x4_t n_sum = vmulq_f32(n_row_v[0], n_filter_v[0]);
            n_sum = vmlaq_f32(n_sum, n_row_v[1], n_filter_v[1]);
            n_sum = vmlaq_f32(n_sum, n_row_v[2], n_filter_v[2]);
            n_sum = vsetq_lane_f32(*output_ptr_v[0], n_sum, 3);
            *output_ptr_v[0] = vaddvq_f32(n_sum);

            ++output_ptr_v[0];
            for (int i = 0; i < 3; ++i) {
              row_ptr_v[i] += 1;
            }
          }
        }

        filter_ptr += kFilterSize;
        input_ptr += input_height * input_width;
      }
    }
  }
}

void Conv2dNeonK3x3S2(const float *input,  // NCHW
                      const index_t *input_shape,
                      const float *filter,  // c_out, c_in, kernel_h, kernel_w
                      const index_t *filter_shape,
                      const float *bias,  // c_out
                      float *output,      // NCHW
                      const index_t *output_shape) {
  int tail_step = 2 * (input_shape[3] - output_shape[3]);

  int output_batch = output_shape[0];
  int output_channels = output_shape[1];
  int output_height = output_shape[2];
  int output_width = output_shape[3];
  int input_batch = input_shape[0];
  int input_channels = input_shape[1];
  int input_height = input_shape[2];
  int input_width = input_shape[3];
  int multiplier =
      filter_shape == nullptr ? 0 : filter_shape[0];
  int filter_in_channels =
      filter_shape == nullptr ? input_channels : 1;

#pragma omp parallel for collapse(2)
  for (int b = 0; b < output_batch; ++b) {
    for (int oc = 0; oc < output_channels; ++oc) {
      float *output_ptr_base =
          output + b * output_channels * output_height * output_width;
      const float *filter_ptr = filter + oc * filter_in_channels * kFilterSize;
      const float *input_ptr =
          input + b * input_channels * input_height * input_width;
      if (filter_shape != nullptr) {
        input_ptr += (oc / multiplier) * input_height * input_width;
      }
      float *output_ptr = output_ptr_base + oc * output_height * output_width;
      std::fill(output_ptr, output_ptr + output_height * output_width,
                bias ? bias[oc] : 0);
      for (int ic = 0; ic < filter_in_channels; ++ic) {
        float32x4_t n_filter_v[3] = {vld1q_f32(filter_ptr),
                                     vld1q_f32(filter_ptr + 3),
                                     vld1q_f32(filter_ptr + 6)};

        const float *row_ptr_v[3] = {input_ptr, input_ptr + input_width,
                                     input_ptr + 2 * input_width};

        float *output_ptr_inner = output_ptr;

        for (int h = 0; h < output_height; ++h) {
          int count = output_width >> 2;
          int remain_count = output_width & 3;

          for (; count > 0; --count) {
            float32x4_t n_sum = vdupq_n_f32(.0f);

            float32x4x2_t n_row_former = vld2q_f32(row_ptr_v[0]);
            float32x4_t n_row_latter = vld1q_f32(row_ptr_v[0] + 8);
            float32x4_t n_row_ext =
                vextq_f32(n_row_former.val[0], n_row_latter, 1);

            n_sum =
                vfmaq_laneq_f32(n_sum, n_row_former.val[0], n_filter_v[0], 0);
            n_sum =
                vfmaq_laneq_f32(n_sum, n_row_former.val[1], n_filter_v[0], 1);
            n_sum = vfmaq_laneq_f32(n_sum, n_row_ext, n_filter_v[0], 2);

            float32x4x2_t n_row1_former = vld2q_f32(row_ptr_v[1]);
            float32x4_t n_row1_latter = vld1q_f32(row_ptr_v[1] + 8);
            float32x4_t n_row1_ext =
                vextq_f32(n_row1_former.val[0], n_row1_latter, 1);
            n_sum =
                vfmaq_laneq_f32(n_sum, n_row1_former.val[0], n_filter_v[1], 0);
            n_sum =
                vfmaq_laneq_f32(n_sum, n_row1_former.val[1], n_filter_v[1], 1);
            n_sum = vfmaq_laneq_f32(n_sum, n_row1_ext, n_filter_v[1], 2);

            float32x4x2_t n_row2_former = vld2q_f32(row_ptr_v[2]);
            float32x4_t n_row2_latter = vld1q_f32(row_ptr_v[2] + 8);
            float32x4_t n_row2_ext =
                vextq_f32(n_row2_former.val[0], n_row2_latter, 1);
            n_sum =
                vfmaq_laneq_f32(n_sum, n_row2_former.val[0], n_filter_v[2], 0);
            n_sum =
                vfmaq_laneq_f32(n_sum, n_row2_former.val[1], n_filter_v[2], 1);
            n_sum = vfmaq_laneq_f32(n_sum, n_row2_ext, n_filter_v[2], 2);

            float32x4_t n_output_row = vld1q_f32(output_ptr_inner);
            n_output_row = vaddq_f32(n_output_row, n_sum);
            vst1q_f32(output_ptr_inner, n_output_row);
            output_ptr_inner += kRegisterSize;
            for (int i = 0; i < 3; ++i) {
              row_ptr_v[i] += 2 * kRegisterSize;
            }
          }
          for (; remain_count > 0; --remain_count) {
            float32x4_t n_row_v[] = {vld1q_f32(row_ptr_v[0]),
                                     vld1q_f32(row_ptr_v[1]),
                                     vld1q_f32(row_ptr_v[2])};
            float32x4_t n_sum = vmulq_f32(n_row_v[0], n_filter_v[0]);
            n_sum = vmlaq_f32(n_sum, n_row_v[1], n_filter_v[1]);
            n_sum = vmlaq_f32(n_sum, n_row_v[2], n_filter_v[2]);
            n_sum = vsetq_lane_f32(*output_ptr_inner, n_sum, 3);
            *output_ptr_inner = vaddvq_f32(n_sum);

            ++output_ptr_inner;
            for (int i = 0; i < 3; ++i) {
              row_ptr_v[i] += 2;
            }
          }
          for (int i = 0; i < 3; ++i) {
            row_ptr_v[i] += tail_step;
          }
        }

        filter_ptr += kFilterSize;
        input_ptr += input_height * input_width;
      }
    }
  }
}
}  //  namespace kernels
}  //  namespace mace
