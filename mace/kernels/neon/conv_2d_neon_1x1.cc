//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <arm_neon.h>
#include "mace/core/common.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {
static constexpr index_t kInputChannelBlockSize = 2;
static constexpr index_t kOutputChannelBlockSize = 4;
static __attribute__((__aligned__(64))) int32_t mask_array[8] = {
    0, 0, 0, 0, -1, -1, -1, -1
};

static inline void NeonConv2x4Kernel(index_t input_channels,
                                     index_t pixel_size,
                                     const float *input,
                                     const float *filter,
                                     float *output) {
  const float *input0 = input;
  const float *input1 = input + pixel_size;

  const float32x2_t vfilter0x = vld1_f32(filter);
  filter += input_channels;
  const float32x2_t vfilter1x = vld1_f32(filter);
  filter += input_channels;
  const float32x2_t vfilter2x = vld1_f32(filter);
  filter += input_channels;
  const float32x2_t vfilter3x = vld1_f32(filter);

  float *output0 = output;
  float *output1 = output0 + pixel_size;
  float *output2 = output1 + pixel_size;
  float *output3 = output2 + pixel_size;
  while (pixel_size >= 4) {
    float32x4_t voutput0 = vld1q_f32(output0);
    float32x4_t voutput1 = vld1q_f32(output1);
    float32x4_t voutput2 = vld1q_f32(output2);
    float32x4_t voutput3 = vld1q_f32(output3);

    const float32x4_t vinput0 = vld1q_f32(input0);
    input0 += 4;
    voutput0 = vfmaq_lane_f32(voutput0, vinput0, vfilter0x, 0);
    voutput1 = vfmaq_lane_f32(voutput1, vinput0, vfilter1x, 0);
    voutput2 = vfmaq_lane_f32(voutput2, vinput0, vfilter2x, 0);
    voutput3 = vfmaq_lane_f32(voutput3, vinput0, vfilter3x, 0);

    const float32x4_t vinput1 = vld1q_f32(input1);
    input1 += 4;
    voutput0 = vfmaq_lane_f32(voutput0, vinput1, vfilter0x, 1);
    voutput1 = vfmaq_lane_f32(voutput1, vinput1, vfilter1x, 1);
    voutput2 = vfmaq_lane_f32(voutput2, vinput1, vfilter2x, 1);
    voutput3 = vfmaq_lane_f32(voutput3, vinput1, vfilter3x, 1);

    vst1q_f32(output0, voutput0);
    output0 += 4;
    vst1q_f32(output1, voutput1);
    output1 += 4;
    vst1q_f32(output2, voutput2);
    output2 += 4;
    vst1q_f32(output3, voutput3);
    output3 += 4;

    pixel_size -= 4;
  }
  if (pixel_size != 0) {
    const int32x4_t vmask = vld1q_s32(&mask_array[pixel_size]);

    output0 = output0 + pixel_size - 4;
    float32x4_t voutput0 = vld1q_f32(output0);
    output1 = output1 + pixel_size - 4;
    float32x4_t voutput1 = vld1q_f32(output1);
    output2 = output2 + pixel_size - 4;
    float32x4_t voutput2 = vld1q_f32(output2);
    output3 = output3 + pixel_size - 4;
    float32x4_t voutput3 = vld1q_f32(output3);

    const float32x4_t vinput0 = vreinterpretq_f32_s32(
        vandq_s32(vmask, vreinterpretq_s32_f32(vld1q_f32(&input0[pixel_size - 4]))));
    voutput0 = vfmaq_lane_f32(voutput0, vinput0, vfilter0x, 0);
    voutput1 = vfmaq_lane_f32(voutput1, vinput0, vfilter1x, 0);
    voutput2 = vfmaq_lane_f32(voutput2, vinput0, vfilter2x, 0);
    voutput3 = vfmaq_lane_f32(voutput3, vinput0, vfilter3x, 0);

    const float32x4_t vinput1 = vreinterpretq_f32_s32(
        vandq_s32(vmask, vreinterpretq_s32_f32(vld1q_f32(&input1[pixel_size - 4]))));
    voutput0 = vfmaq_lane_f32(voutput0, vinput1, vfilter0x, 1);
    voutput1 = vfmaq_lane_f32(voutput1, vinput1, vfilter1x, 1);
    voutput2 = vfmaq_lane_f32(voutput2, vinput1, vfilter2x, 1);
    voutput3 = vfmaq_lane_f32(voutput3, vinput1, vfilter3x, 1);

    vst1q_f32(output0, voutput0);
    vst1q_f32(output1, voutput1);
    vst1q_f32(output2, voutput2);
    vst1q_f32(output3, voutput3);
  }
}

static inline void NeonConv2x4SubBlockKernel(index_t input_channels_subblock_size,
                                             index_t output_channels_subblock_size,
                                             index_t input_channels,
                                             index_t pixel_size,
                                             const float *input,
                                             const float *filter,
                                             float *output) {
  const float *input0 = input;
  const float *input1 = input + pixel_size;

  float32x2_t vfilter0x, vfilter1x, vfilter2x, vfilter3x;
  vfilter0x = vld1_dup_f32(&filter[0]);
  if (input_channels_subblock_size > 1) {
    vfilter0x = vld1_lane_f32(&filter[1], vfilter0x, 1);
  }
  if (output_channels_subblock_size > 1) {
    filter += input_channels;
    vfilter1x = vld1_dup_f32(&filter[0]);
    if (input_channels_subblock_size > 1) {
      vfilter1x = vld1_lane_f32(&filter[1], vfilter1x, 1);
    }
    if (output_channels_subblock_size > 2) {
      filter += input_channels;
      vfilter2x = vld1_dup_f32(&filter[0]);
      if (input_channels_subblock_size > 1) {
        vfilter2x = vld1_lane_f32(&filter[1], vfilter2x, 1);
      }
      if (output_channels_subblock_size > 3) {
        filter += input_channels;
        vfilter3x = vld1_dup_f32(&filter[0]);
        if (input_channels_subblock_size > 1) {
          vfilter3x = vld1_lane_f32(&filter[1], vfilter3x, 1);
        }
      }
    }
  }

  float *output0 = output;
  float *output1 = output0 + pixel_size;
  float *output2 = output1 + pixel_size;
  float *output3 = output2 + pixel_size;
  while (pixel_size >= 4) {
    float32x4_t voutput0, voutput1, voutput2, voutput3;
    voutput0 = vld1q_f32(output0);
    if (output_channels_subblock_size > 1) {
      voutput1 = vld1q_f32(output1);
      if (output_channels_subblock_size > 2) {
        voutput2 = vld1q_f32(output2);
        if (output_channels_subblock_size > 3) {
          voutput3 = vld1q_f32(output3);
        }
      }
    }

    const float32x4_t vinput0 = vld1q_f32(input0);
    input0 += 4;
    voutput0 = vfmaq_lane_f32(voutput0, vinput0, vfilter0x, 0);
    voutput1 = vfmaq_lane_f32(voutput1, vinput0, vfilter1x, 0);
    voutput2 = vfmaq_lane_f32(voutput2, vinput0, vfilter2x, 0);
    voutput3 = vfmaq_lane_f32(voutput3, vinput0, vfilter3x, 0);

    if (input_channels_subblock_size > 1) {
      const float32x4_t vinput1 = vld1q_f32(input1);
      input1 += 4;
      voutput0 = vfmaq_lane_f32(voutput0, vinput1, vfilter0x, 1);
      voutput1 = vfmaq_lane_f32(voutput1, vinput1, vfilter1x, 1);
      voutput2 = vfmaq_lane_f32(voutput2, vinput1, vfilter2x, 1);
      voutput3 = vfmaq_lane_f32(voutput3, vinput1, vfilter3x, 1);
    }

    vst1q_f32(output0, voutput0);
    output0 += 4;
    if (output_channels_subblock_size > 1) {
      vst1q_f32(output1, voutput1);
      output1 += 4;
      if (output_channels_subblock_size > 2) {
        vst1q_f32(output2, voutput2);
        output2 += 4;
        if (output_channels_subblock_size > 3) {
          vst1q_f32(output3, voutput3);
          output3 += 4;
        }
      }
    }

    pixel_size -= 4;
  }
  if (pixel_size != 0) {
    const int32x4_t vmask = vld1q_s32(&mask_array[pixel_size]);

    float32x4_t voutput0, voutput1, voutput2, voutput3;
    output0 += pixel_size - 4;
    voutput0 = vld1q_f32(output0);
    if (output_channels_subblock_size > 1) {
      output1 += pixel_size - 4;
      voutput1 = vld1q_f32(output1);
      if (output_channels_subblock_size > 2) {
        output2 += pixel_size - 4;
        voutput2 = vld1q_f32(output2);
        if (output_channels_subblock_size > 3) {
          output3 += pixel_size - 4;
          voutput3 = vld1q_f32(output3);
        }
      }
    }

    const float32x4_t vinput0 = vreinterpretq_f32_s32(
        vandq_s32(vmask, vreinterpretq_s32_f32(vld1q_f32(&input0[pixel_size - 4]))));
    voutput0 = vfmaq_lane_f32(voutput0, vinput0, vfilter0x, 0);
    voutput1 = vfmaq_lane_f32(voutput1, vinput0, vfilter1x, 0);
    voutput2 = vfmaq_lane_f32(voutput2, vinput0, vfilter2x, 0);
    voutput3 = vfmaq_lane_f32(voutput3, vinput0, vfilter3x, 0);

    if (input_channels_subblock_size > 1) {
      const float32x4_t vinput1 = vreinterpretq_f32_s32(
          vandq_s32(vmask, vreinterpretq_s32_f32(vld1q_f32(&input1[pixel_size - 4]))));
      voutput0 = vfmaq_lane_f32(voutput0, vinput1, vfilter0x, 1);
      voutput1 = vfmaq_lane_f32(voutput1, vinput1, vfilter1x, 1);
      voutput2 = vfmaq_lane_f32(voutput2, vinput1, vfilter2x, 1);
      voutput3 = vfmaq_lane_f32(voutput3, vinput1, vfilter3x, 1);
    }

    vst1q_f32(output0, voutput0);
    if (output_channels_subblock_size > 1) {
      vst1q_f32(output1, voutput1);
      if (output_channels_subblock_size > 2) {
        vst1q_f32(output2, voutput2);
        if (output_channels_subblock_size > 3) {
          vst1q_f32(output3, voutput3);
        }
      }
    }
  }
}

void Conv2dNeonK1x1S1(const float *input,  // NCHW
                      const index_t *input_shape,
                      const float *filter,  // c_out, c_in, filter_h, filter_w
                      const index_t *filter_shape,
                      const float *bias,    // c_out
                      float *output,        // NCHW
                      const index_t *output_shape) {
  const index_t batch = output_shape[0];
  const index_t channels = output_shape[1];
  const index_t height = output_shape[2];
  const index_t width = output_shape[3];

  const index_t input_batch = input_shape[0];
  const index_t input_channels = input_shape[1];
  const index_t input_height = input_shape[2];
  const index_t input_width = input_shape[3];

  MACE_CHECK(input_batch == batch && input_height == height &&
      input_width == width);

  const index_t total_pixels = height * width;
  const index_t round_up_channels = RoundUp(channels, kOutputChannelBlockSize);

#pragma omp parallel for collapse(2)
  for (index_t n = 0; n < batch; ++n) {
    for (int i = 0; i < channels; ++i) {
      float *output_ptr_base = output + n * channels * total_pixels + i * total_pixels;
      std::fill(output_ptr_base, output_ptr_base + total_pixels, bias ? bias[i] : 0);
    }
  }
  // benchmark omp collapsed(2)
#pragma omp parallel for collapse(2)
  for (index_t n = 0; n < batch; ++n) {
    for (index_t c = 0; c < round_up_channels; c += kOutputChannelBlockSize) {
      const float *input_ptr = input + n * input_channels * total_pixels;
      const float *filter_ptr = filter + c * input_channels;
      float *output_ptr = output + n * channels * total_pixels + c * total_pixels;
      const index_t output_channel_block_size = std::min(channels - c, kOutputChannelBlockSize);
      index_t remain_input_channels = input_channels;
      if (c + kOutputChannelBlockSize <= channels) {
        while (remain_input_channels >= kInputChannelBlockSize) {
          NeonConv2x4Kernel(input_channels, total_pixels, input_ptr, filter_ptr, output_ptr);

          input_ptr += kInputChannelBlockSize * total_pixels;
          filter_ptr += kInputChannelBlockSize;
          remain_input_channels -= kInputChannelBlockSize;
        }
      }
      while (remain_input_channels != 0) {
        const index_t input_channel_block_size = std::min(remain_input_channels, kInputChannelBlockSize);
        NeonConv2x4SubBlockKernel(input_channel_block_size, output_channel_block_size,
                                  input_channels, total_pixels, input_ptr, filter_ptr, output_ptr);
        input_ptr += kInputChannelBlockSize * total_pixels;
        filter_ptr += kInputChannelBlockSize;
        remain_input_channels -= input_channel_block_size;
      }

    }
  }
};

void Conv2dNeonPixelK1x1S1(const float *input,  // NCHW
                      const index_t *input_shape,
                      const float *filter,  // c_out, c_in, kernel_h, kernel_w
                      const index_t *filter_shape,
                      const float *bias,    // c_out
                      float *output,        // NCHW
                      const index_t *output_shape) {
  const index_t batch = output_shape[0];
  const index_t channels = output_shape[1];
  const index_t height = output_shape[2];
  const index_t width = output_shape[3];

  const index_t input_batch = input_shape[0];
  const index_t input_channels = input_shape[1];
  const index_t input_height = input_shape[2];
  const index_t input_width = input_shape[3];

  MACE_CHECK(input_batch == batch && input_height == height &&
      input_width == width);

  const index_t total_pixels = height * width;
  // Process 4 * 2 = 8 pixels for each innermost loop
  // TODO Does 64 bit v.s. 32 bit index matters? need benchmark
  const index_t total_loops = total_pixels >> 3;
  const index_t loop_remaining = total_pixels & 7;

  // benchmark omp collapsed(2)
#pragma omp parallel for collapse(2)
  for (index_t n = 0; n < batch; ++n) {
    for (index_t c = 0; c < channels; ++c) {
      const float *filter_ptr = filter + c * input_channels;
      // TODO Will GCC opt these out?
      float *channel_output_start =
          output + n * channels * height * width + c * height * width;
      const float *input_ptr =
          input + n * input_channels * input_height * input_width;

      // Fill with bias
      float *output_ptr = channel_output_start;
      std::fill(output_ptr, output_ptr + total_pixels, bias ? bias[c] : 0);

      index_t inc = 0;
      // Process 4 input channels in batch
      for (; inc + 3 < input_channels; inc += 4) {
        float *output_ptr = channel_output_start;
        // The begining of each input feature map channel
        MACE_ASSERT(input_ptr ==
            input + n * input_channels * input_height * input_width +
                inc * input_height * input_width);

        const float *input_ptr1 = input_ptr + total_pixels;
        const float *input_ptr2 = input_ptr1 + total_pixels;
        const float *input_ptr3 = input_ptr2 + total_pixels;

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
          input_ptr += 8;
          input_ptr1 += 8;
          input_ptr2 += 8;
          input_ptr3 += 8;
        }
        // Process the remaining pixels
        index_t remaining_pixels = loop_remaining;
        for (; remaining_pixels > 0; --remaining_pixels) {
          const float mul = *input_ptr * k0;
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
        float *output_ptr = channel_output_start;
        MACE_ASSERT(input_ptr ==
            input + n * input_channels * input_height * input_width +
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
          input_ptr += 8;
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

}  // namespace kernels
}  // namespace mace
