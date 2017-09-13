//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <arm_neon.h>
#include "mace/kernels/conv_2d.h"

namespace mace {
namespace kernels {

void Conv2dNeonK1x1S1(const float* input, // NCHW
                      const index_t* input_shape,
                      const float* filter, // c_out, c_in, kernel_h, kernel_w
                      const float* bias, // c_out
                      float* output, // NCHW
                      const index_t* output_shape) {
  const index_t batch    = output_shape[0];
  const index_t channels = output_shape[1];
  const index_t height   = output_shape[2];
  const index_t width    = output_shape[3];

  const index_t input_batch    = input_shape[0];
  const index_t input_channels = input_shape[1];
  const index_t input_height   = input_shape[2];
  const index_t input_width    = input_shape[3];

  MACE_CHECK(input_batch  == batch &&
             input_height == height &&
             input_width  == width);

  const index_t total_pixels = height * width;
  // Process 4 * 2 = 8 pixels for each innermost loop
  // TODO Does 64 bit v.s. 32 bit index matters? need benchmark
  const index_t total_loops = total_pixels >> 3;
  const index_t loop_remaining = total_pixels & 7;

  // benchmark omp collapsed(2)
  for (index_t n = 0; n < batch; ++n) {
    const float* filter_ptr = filter;
    #pragma omp parallel for
    for (index_t c = 0; c < channels; ++c) {
      // TODO Will GCC opt these out?
      float* channel_output_start =
        output + n * channels * height * width + c * height * width;
      const float* input_ptr = input + n * input_channels * input_height * input_width;

      // Fill with bias
      float* output_ptr = channel_output_start;
      for (index_t ptr = 0; ptr < total_pixels; ++ptr) {
        output_ptr[ptr] = bias[c]; // TODO can we avoid this?
      }

      index_t inc = 0;
      // Process 4 input channels in batch
      for (; inc + 3 < input_channels; inc += 4) {
        float* output_ptr = channel_output_start;
        // The begining of each input feature map channel
        MACE_ASSERT(input_ptr == input + n * input_channels *
                                         input_height * input_width +
                                 inc * input_height * input_width);

        const float* input_ptr1 = input_ptr  + total_pixels;
        const float* input_ptr2 = input_ptr1 + total_pixels;
        const float* input_ptr3 = input_ptr2 + total_pixels;


        // filter is in c_out, c_in, 1, 1 order
        MACE_ASSERT(filter_ptr == filter + c * input_channels + inc);
        const float k0 = filter_ptr[0];
        const float k1 = filter_ptr[1];
        const float k2 = filter_ptr[2];
        const float k3 = filter_ptr[3];
        filter_ptr += 4;

        const float32x4_t vk0 = vdupq_n_f32(k0);
        const float32x4_t vk1 = vdupq_n_f32(k1);
        const float32x4_t vk2 = vdupq_n_f32(k2);
        const float32x4_t vk3 = vdupq_n_f32(k3);

        index_t loop_itr = total_loops;
        for (; loop_itr > 0; --loop_itr) {
          // Process 2 group of 4 floats
          float32x4_t out0 = vld1q_f32(output_ptr);
          float32x4_t out4 = vld1q_f32(output_ptr + 4);

          const float32x4_t in00 = vld1q_f32(input_ptr);
          const float32x4_t in04 = vld1q_f32(input_ptr + 4);

          out0 = vfmaq_f32(out0, in00, vk0);
          out4 = vfmaq_f32(out4, in04, vk0);

          const float32x4_t in10 = vld1q_f32(input_ptr1);
          const float32x4_t in14 = vld1q_f32(input_ptr1 + 4);

          out0 = vfmaq_f32(out0, in10, vk1);
          out4 = vfmaq_f32(out4, in14, vk1);

          const float32x4_t in20 = vld1q_f32(input_ptr2);
          const float32x4_t in24 = vld1q_f32(input_ptr2 + 4);

          out0 = vfmaq_f32(out0, in20, vk2);
          out4 = vfmaq_f32(out4, in24, vk2);

          const float32x4_t in30 = vld1q_f32(input_ptr3);
          const float32x4_t in34 = vld1q_f32(input_ptr3 + 4);

          out0 = vfmaq_f32(out0, in30, vk3);
          out4 = vfmaq_f32(out4, in34, vk3);

          float prev_output = output_ptr[0];
          // Save output
          vst1q_f32(output_ptr, out0);
          vst1q_f32(output_ptr + 4, out4);

          output_ptr += 8;
          input_ptr  += 8;
          input_ptr1 += 8;
          input_ptr2 += 8;
          input_ptr3 += 8;
        }
        // Process the remaining pixels
        index_t remaining_pixels = loop_remaining;
        for (; remaining_pixels > 0; --remaining_pixels) {
          const float mul  = *input_ptr  * k0;
          const float mul1 = *input_ptr1 * k1;
          const float mul2 = *input_ptr2 * k2;
          const float mul3 = *input_ptr3 * k3;

          float prev_output = output_ptr[0];
          *output_ptr += mul + mul1 + mul2 + mul3;

          ++output_ptr;
          ++input_ptr;
          ++input_ptr1;
          ++input_ptr2;
          ++input_ptr3;
        }
        // Skip these 4 feature maps
        input_ptr += 3 * total_pixels;
      }
      // Process the remaining channels
      for (; inc < input_channels; ++inc) {
        float* output_ptr = channel_output_start;
        MACE_ASSERT(input_ptr == input + n * input_channels *
                                         input_height * input_width +
                                 inc * input_height * input_width);
        MACE_ASSERT(filter_ptr == filter + c * input_channels + inc);

        const float k0 = filter_ptr[0];
        ++filter_ptr;
        const float32x4_t vk0 = vdupq_n_f32(k0);

        index_t loop_itr = total_loops;
        for (; loop_itr > 0; --loop_itr) {
          float32x4_t out0 = vld1q_f32(output_ptr);
          float32x4_t out4 = vld1q_f32(output_ptr + 4);

          const float32x4_t in0 = vld1q_f32(input_ptr);
          const float32x4_t in4 = vld1q_f32(input_ptr + 4);

          out0 = vfmaq_f32(out0, in0, vk0);
          out4 = vfmaq_f32(out4, in4, vk0);

          // Save output
          vst1q_f32(output_ptr, out0);
          vst1q_f32(output_ptr + 4, out4);

          output_ptr += 8;
          input_ptr  += 8;
        }
        // Process the remaining pixels
        index_t remaining_pixels = loop_remaining;
        for (; remaining_pixels > 0; --remaining_pixels) {
          const float mul = *input_ptr * k0;
          
          *output_ptr += mul;

          ++output_ptr;
          ++input_ptr;
        }
      }
    }
  }
};

} // namespace kernels
} // namespace mace
