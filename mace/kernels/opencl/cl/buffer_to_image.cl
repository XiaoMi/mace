#include <common.h>

__kernel void filter_buffer_to_image(__global const DATA_TYPE *input, /* h, w, ic, oc */
                                     __private const int filter_w,
                                     __private const int in_channel,
                                     __private const int out_channel,
                                     __write_only image2d_t output) {
  int w = get_global_id(0);
  int h = get_global_id(1);
  const int out_channel_idx = h * 4;
  const int rounded_in_channel = ((in_channel + 3) / 4) * 4;
  const int hw_idx = w / rounded_in_channel;
  const int in_channel_idx = w % rounded_in_channel;
  const int h_idx = hw_idx / filter_w;
  const int w_idx = hw_idx % filter_w;
  const int offset = ((h_idx * filter_w + w_idx) * in_channel + in_channel_idx) * out_channel
                           + out_channel_idx;

  const int size = out_channel - out_channel_idx;
  VEC_DATA_TYPE(DATA_TYPE, 4) values = 0;
  if (in_channel_idx < in_channel) {
    if (size < 4) {
      switch(size) {
        case 3:
          values.z = *(input + offset + 2);
        case 2:
          values.y = *(input + offset + 1);
        case 1:
          values.x = *(input + offset);
      }
    } else {
      values = vload4(0, input + offset);
    }
  }

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
  const int rounded_in_channel = ((in_channel + 3) / 4) * 4;
  const int hw_idx = w / rounded_in_channel;
  const int in_channel_idx = w % rounded_in_channel;
  const int h_idx = hw_idx / filter_w;
  const int w_idx = hw_idx % filter_w;
  const int offset = ((h_idx * filter_w + w_idx) * in_channel + in_channel_idx) * out_channel
                           + out_channel_idx;

  if (in_channel_idx < in_channel) {
    int2 coord = (int2)(w, h);
    VEC_DATA_TYPE(DATA_TYPE, 4) values = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, SAMPLER, coord);
    const int size = (out_channel - out_channel_idx);
    if (size < 4) {
      switch (size) {
        case 3:
          output[offset+2] = values.s2;
        case 2:
          output[offset+1] = values.s1;
        case 1:
          output[offset] = values.s0;
      }
    } else {
      vstore4(values, 0, output + offset);
    }
  }
}


__kernel void dw_filter_buffer_to_image(__global const DATA_TYPE *input, /* h, w, ic, m */
                                        __private const int filter_w,
                                        __private const int in_channel,
                                        __private const int multiplier,
                                        __write_only image2d_t output) { /* ic%4 * kh * kw * m, ic/4 */
  const int w = get_global_id(0);
  const int h = get_global_id(1);

  DATA_TYPE4 values = 0;
  if (multiplier == 1) {
    const int in_channel_idx = h << 2;
    const int h_idx = w / filter_w;
    const int w_idx = w % filter_w;

    const int offset = mad24(mad24(h_idx, filter_w, w_idx),
                             in_channel, in_channel_idx);

    const int size = in_channel - in_channel_idx;
    if (in_channel_idx < in_channel) {
      if (size < 4) {
        switch(size) {
          case 3:
            values.z = *(input + offset + 2);
          case 2:
            values.y = *(input + offset + 1);
          case 1:
            values.x = *(input + offset);
        }
      } else {
        values = vload4(0, input + offset);
      }
    }
  } else {
    const int in_channel_idx = h << 2;
    const int m = w % multiplier;
    const int hw_idx = w / multiplier;
    const int h_idx = hw_idx / filter_w;
    const int w_idx = hw_idx % filter_w;

    const int offset = mad24(mad24(mad24(h_idx, filter_w, w_idx),
                in_channel, in_channel_idx),
            multiplier, m);
    // TODO support multiplier > 1
  }

  int2 coord = (int2)(w, h);
  WRITE_IMAGET(output, coord, values);
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

  const int size = channels - channel_idx;
  VEC_DATA_TYPE(DATA_TYPE, 4) values = 0;
  if (size < 4) {
    switch(size) {
      case 3:
        values.z = *(input + offset + 2);
      case 2:
        values.y = *(input + offset + 1);
      case 1:
        values.x = *(input + offset);
    }
  } else {
    values = vload4(0, input + offset);
  }
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

  int2 coord = (int2)(w, h);
  VEC_DATA_TYPE(DATA_TYPE, 4) values = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, SAMPLER, coord);
  const int size = channels - channel_idx;
  if (size < 4) {
    switch (size) {
      case 3:
        output[offset+2] = values.s2;
      case 2:
        output[offset+1] = values.s1;
      case 1:
        output[offset] = values.s0;
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

  const int size = count - offset;
  VEC_DATA_TYPE(DATA_TYPE, 4) values = 0;
  if (size < 4) {
    switch(size) {
      case 3:
        values.z = *(input + offset + 2);
      case 2:
        values.y = *(input + offset + 1);
      case 1:
        values.x = *(input + offset);
    }
  } else {
    values = vload4(0, input + offset);
  }
  int2 coord = (int2)(w, h);
  CMD_TYPE(write_image, CMD_DATA_TYPE)(output, coord, values);
}

__kernel void arg_image_to_buffer(__global DATA_TYPE *output, /* nhwc */
                                  __private const int count,
                                  __read_only image2d_t input) {
  int w = get_global_id(0);
  int h = get_global_id(1);
  const int offset = w * 4;

  int2 coord = (int2)(w, h);
  VEC_DATA_TYPE(DATA_TYPE, 4) values = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, SAMPLER, coord);
  const int size = count - offset;
  if (size < 4) {
    switch (size) {
      case 3:
        output[offset+2] = values.s2;
      case 2:
        output[offset+1] = values.s1;
      case 1:
        output[offset] = values.s0;
    }
  } else {
    vstore4(values, 0, output + offset);
  }
}
