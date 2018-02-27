//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <arm_neon.h>
#include <float.h>
#include <limits>

namespace mace {
namespace kernels {

void PoolingAvgNeonK2x2S2x2(const float *input,
                            const index_t *in_shape,
                            float *output,
                            const index_t *out_shape,
                            const int *paddings) {
  index_t batch = in_shape[0];
  index_t channels = in_shape[1];
  index_t in_height = in_shape[2];
  index_t in_width = in_shape[3];

  index_t out_height = out_shape[2];
  index_t out_width = out_shape[3];

  int padding_top = paddings[0] / 2;
  int padding_bottom = paddings[0] - padding_top;
  int padding_left = paddings[1] / 2;
  int padding_right = paddings[1] - padding_left;

  int in_image_size = in_height * in_width;
  int out_image_size = out_height * out_width;
  index_t input_offset = 0;
  index_t output_offset = 0;
  float avg_factors[4] = {0.25, 0.25, 0.25, 0.25};

#pragma omp parallel for collapse(2)
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      float *outptr = output + output_offset;
      const float *r0, *r1;

      for (int h = 0; h < out_height; ++h) {
        int w = 0;
        int num_vectors = 0;
        if (!((h == 0 && padding_top > 0) ||
              (h == out_height - 1 && padding_bottom > 0))) {
          r0 = input + input_offset + (h * 2 - padding_top) * in_width;
          r1 = r0 + in_width;
          if (padding_left > 0) {
            *outptr = (r0[0] + r1[0]) * 0.25;
            ++r0;
            ++r1;
            ++outptr;
            ++w;
          }
          if (padding_right > 0) {
            num_vectors = (out_width - w - 1) >> 2;
          } else {
            num_vectors = (out_width - w) >> 2;
          }
        }

        w += num_vectors << 2;
        float32x4_t factors = vld1q_f32(avg_factors);
        for (; num_vectors > 0; --num_vectors) {
          float32x4_t r00 = vld1q_f32(r0);
          float32x4_t r10 = vld1q_f32(r1);
          float32x4_t r01 = vld1q_f32(r0 + 4);
          float32x4_t r11 = vld1q_f32(r1 + 4);

          float32x4_t sum0 = vaddq_f32(r00, r10);
          float32x4_t sum1 = vaddq_f32(r01, r11);

          float32x4_t sum_result = vpaddq_f32(sum0, sum1);
          float32x4_t avg_result = vmulq_f32(sum_result, factors);

          vst1q_f32(outptr, avg_result);

          r0 += 8;
          r1 += 8;
          outptr += 4;
        }

        for (; w < out_width; ++w) {
          float sum = 0.0;
          for (int kh = 0; kh < 2; ++kh) {
            for (int kw = 0; kw < 2; ++kw) {
              int inh = h * 2 - padding_top + kh;
              int inw = w * 2 - padding_left + kw;
              if (inh >= 0 && inh < in_height && inw >= 0 && inw < in_width) {
                sum += input[input_offset + inh * in_width + inw];
              }
            }
          }

          *outptr = sum * 0.25;
          ++outptr;
        }
      }
      input_offset += in_image_size;
      output_offset += out_image_size;
    }
  }
}

// assume the input has already been padded
void PoolingAvgNeonK2x2S2x2Padded(const float *input,
                                  const index_t *in_shape,
                                  float *output,
                                  const index_t *out_shape) {
  index_t batch = in_shape[0];
  index_t channels = in_shape[1];
  index_t in_height = in_shape[2];
  index_t in_width = in_shape[3];

  index_t out_height = out_shape[2];
  index_t out_width = out_shape[3];

  int in_image_size = in_height * in_width;
  int out_image_size = out_height * out_width;
  index_t input_offset = 0;
  index_t output_offset = 0;
  float avg_factors[4] = {0.25, 0.25, 0.25, 0.25};

#pragma omp parallel for collapse(2)
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channels; ++c) {
      const float *img0 = input + input_offset;
      float *outptr = output + output_offset;

      const float *r0 = img0;
      const float *r1 = img0 + in_width;

      for (int h = 0; h < out_height; ++h) {
        int num_vectors = out_width >> 2;
        int remain = out_width - (num_vectors << 2);

        float32x4_t factors = vld1q_f32(avg_factors);
        for (; num_vectors > 0; --num_vectors) {
          float32x4_t r00 = vld1q_f32(r0);
          float32x4_t r10 = vld1q_f32(r1);
          float32x4_t r01 = vld1q_f32(r0 + 4);
          float32x4_t r11 = vld1q_f32(r1 + 4);

          float32x4_t sum0 = vaddq_f32(r00, r10);
          float32x4_t sum1 = vaddq_f32(r01, r11);

          float32x4_t sum_result = vpaddq_f32(sum0, sum1);
          float32x4_t avg_result = vmulq_f32(sum_result, factors);

          vst1q_f32(outptr, avg_result);

          r0 += 8;
          r1 += 8;
          outptr += 4;
        }

        for (; remain > 0; --remain) {
          *outptr = (r0[0] + r0[1] + r1[0] + r1[1]) * 0.25;

          r0 += 2;
          r1 += 2;
          outptr++;
        }
        r0 += in_width;
        r1 += in_width;
      }
      input_offset += in_image_size;
      output_offset += out_image_size;
    }
  }
}

}  // namespace kernels
}  // namespace mace
