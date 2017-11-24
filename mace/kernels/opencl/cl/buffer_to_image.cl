#include <common.h>

__kernel void filter_buffer_to_image(__global const DATA_TYPE *input, /* h, w, ic, oc */
                                     __private const int filter_w,
                                     __private const int in_channel,
                                     __private const int out_channel,
                                     __write_only image2d_t output) {
  int w = get_global_id(0);
  int h = get_global_id(1);
  const int out_channel_idx = h * 4;
  const int hw_idx = w / in_channel;
  int in_channel_idx = w % in_channel;
  const int h_idx = hw_idx / filter_w;
  const int w_idx = hw_idx % filter_w;
  const int offset = ((h_idx * filter_w + w_idx) * in_channel + in_channel_idx) * out_channel
                           + out_channel_idx;

  VEC_DATA_TYPE(DATA_TYPE, 4) values = vload4(0, input + offset);
  int2 coord = (int2)(w, h);
  CMD_TYPE(write_image, CMD_DATA_TYPE)(output, coord, values);
}

__kernel void filter_image_to_buffer(__global DATA_TYPE *output, /* h, w, ic, oc */
                                     __private const int filter_w,
                                     __private const int in_channel,
                                     __private const int out_channel,
                                     __read_only image2d_t input) {
  int w = get_global_id(0);
  int h = get_global_id(1);
  const int out_channel_idx = h * 4;
  const int hw_idx = w / in_channel;
  int in_channel_idx = w % in_channel;
  const int h_idx = hw_idx / filter_w;
  const int w_idx = hw_idx % filter_w;
  const int offset = ((h_idx * filter_w + w_idx) * in_channel + in_channel_idx) * out_channel
                           + out_channel_idx;

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int2 coord = (int2)(w, h);
  VEC_DATA_TYPE(DATA_TYPE, 4) values = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, sampler, coord);
  if (out_channel_idx + 4 > out_channel) {
    const int diff = in_channel - in_channel_idx;
    output[offset] = values.s0;
    if (diff == 2) {
      output[offset+1] = values.s1;
    } else {
      output[offset+1] = values.s1;
      output[offset+2] = values.s2;
    }
  } else {
    vstore4(values, 0, output + offset);
  }
}

__kernel void in_out_buffer_to_image(__global const DATA_TYPE *input, /* nhwc */
                                     __private const int height,
                                     __private const int width,
                                     __private const int channels,
                                     __write_only image2d_t output) {
  int w = get_global_id(0);
  int h = get_global_id(1);
  const int batch_idx = h / height;
  const int height_idx = h % height;
  const int width_idx = w % width;
  const int channel_idx = w / width * 4;
  const int offset = ((batch_idx * height + height_idx) * width + width_idx) * channels
                           + channel_idx;

  VEC_DATA_TYPE(DATA_TYPE, 4) values = vload4(0, input + offset);
  int2 coord = (int2)(w, h);
  CMD_TYPE(write_image, CMD_DATA_TYPE)(output, coord, values);
}

__kernel void in_out_image_to_buffer(__global DATA_TYPE *output, /* nhwc */
                                     __private const int height,
                                     __private const int width,
                                     __private const int channels,
                                     __read_only image2d_t input) {
  int w = get_global_id(0);
  int h = get_global_id(1);
  const int batch_idx = h / height;
  const int height_idx = h % height;
  const int width_idx = w % width;
  const int channel_idx = w / width * 4;
  const int offset = ((batch_idx * height + height_idx) * width + width_idx) * channels
                           + channel_idx;

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int2 coord = (int2)(w, h);
  VEC_DATA_TYPE(DATA_TYPE, 4) values = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, sampler, coord);
  if (channel_idx + 4 > channels) {
    const int diff = channels - channel_idx;
    output[offset] = values.s0;
    if (diff == 2) {
      output[offset+1] = values.s1;
    } else {
      output[offset+1] = values.s1;
      output[offset+2] = values.s2;
    }
  } else {
    vstore4(values, 0, output + offset);
  }
}

__kernel void arg_buffer_to_image(__global const DATA_TYPE *input, /* nhwc */
                                  __private const int count,
                                  __write_only image2d_t output) {
  int w = get_global_id(0);
  int h = get_global_id(1);
  const int offset = w * 4;

  VEC_DATA_TYPE(DATA_TYPE, 4) values = vload4(0, input + offset);
  int2 coord = (int2)(w, h);
  CMD_TYPE(write_image, CMD_DATA_TYPE)(output, coord, values);
}

__kernel void arg_image_to_buffer(__global DATA_TYPE *output, /* nhwc */
                                  __private const int count,
                                  __read_only image2d_t input) {
  int w = get_global_id(0);
  int h = get_global_id(1);
  const int offset = w * 4;

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int2 coord = (int2)(w, h);
  VEC_DATA_TYPE(DATA_TYPE, 4) values = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, sampler, coord);
  if (offset + 4 > count) {
    const int diff = count - offset;
    output[offset] = values.s0;
    if (diff == 2) {
      output[offset+1] = values.s1;
    } else {
      output[offset+1] = values.s1;
      output[offset+2] = values.s2;
    }
  } else {
    vstore4(values, 0, output + offset);
  }
}
