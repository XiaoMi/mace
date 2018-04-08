#include <common.h>

__kernel void filter_buffer_to_image(KERNEL_ERROR_PARAMS
                                     GLOBAL_WORK_GROUP_SIZE_DIM2
                                     __global const DATA_TYPE *input, /* h, w, oc, ic */
                                     __private const int input_offset,
                                     __private const int filter_h,
                                     __private const int filter_w,
                                     __private const int out_channel,
                                     __private const int in_channel,
                                     __write_only image2d_t output) {
  int w = get_global_id(0);
  int h = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || h >= global_size_dim1) {
    return;
  }
#endif

  const int in_channel_idx = w;
  const int hw_size = filter_w * filter_h;
  const int out_channel_idx = h / hw_size * 4;
  const int hw_idx = h % hw_size;
  const int h_idx = hw_idx / filter_w;
  const int w_idx = hw_idx % filter_w;
  const int offset = input_offset
                     + ((h_idx * filter_w + w_idx) * out_channel
                         + out_channel_idx) * in_channel
                     + in_channel_idx;

  DATA_TYPE4 values = 0;
  if (out_channel_idx < out_channel) {
    const int size = out_channel - out_channel_idx;
    if (size < 4) {
      switch (size) {
        case 3:
          values.z = *(input + offset + 2 * in_channel);
        case 2:
          values.y = *(input + offset + 1 * in_channel);
        case 1:
          values.x = *(input + offset);
      }
    } else {
      values.w = *(input + offset + 3 * in_channel);
      values.z = *(input + offset + 2 * in_channel);
      values.y = *(input + offset + 1 * in_channel);
      values.x = *(input + offset);
    }
  }

  int2 coord = (int2)(w, h);
  WRITE_IMAGET(output, coord, values);
}

__kernel void filter_image_to_buffer(KERNEL_ERROR_PARAMS
                                     GLOBAL_WORK_GROUP_SIZE_DIM2
                                     __global DATA_TYPE *output, /* h, w, oc, ic */
                                     __private const int filter_h,
                                     __private const int filter_w,
                                     __private const int out_channel,
                                     __private const int in_channel,
                                     __read_only image2d_t input) {
  int w = get_global_id(0);
  int h = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || h >= global_size_dim1) {
    return;
  }
#endif

  const int in_channel_idx = w;
  const int hw_size = filter_w * filter_h;
  const int out_channel_idx = h / hw_size * 4;
  const int hw_idx = h % hw_size;
  const int h_idx = hw_idx / filter_w;
  const int w_idx = hw_idx % filter_w;
  const int offset = ((h_idx * filter_w + w_idx) * out_channel
                         + out_channel_idx) * in_channel
                     + in_channel_idx;

  if (out_channel_idx < out_channel) {
    int2 coord = (int2)(w, h);
    DATA_TYPE4 values = READ_IMAGET(input, SAMPLER, coord);
    const int size = (out_channel - out_channel_idx);
    if (size < 4) {
      switch (size) {
        case 3:
          output[offset + 2 * in_channel] = values.z;
        case 2:
          output[offset + 1 * in_channel] = values.y;
        case 1:
          output[offset] = values.x;
      }
    } else {
      output[offset + 3 * in_channel] = values.w;
      output[offset + 2 * in_channel] = values.z;
      output[offset + 1 * in_channel] = values.y;
      output[offset] = values.x;
    }
  }
}

__kernel void dw_filter_buffer_to_image(KERNEL_ERROR_PARAMS
                                        GLOBAL_WORK_GROUP_SIZE_DIM2
                                        __global const DATA_TYPE *input, /* h, w, ic, m */
                                        __private const int input_offset,
                                        __private const int filter_w,
                                        __private const int in_channel,
                                        __private const int multiplier,
                                        __write_only image2d_t output) { /* ic%4 * kh * kw * m, ic/4 */
  const int w = get_global_id(0);
  const int h = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || h >= global_size_dim1) {
    return;
  }
#endif

  DATA_TYPE4 values = 0;
  if (multiplier == 1) {
    const int in_channel_idx = h << 2;
    const int h_idx = w / filter_w;
    const int w_idx = w % filter_w;

    const int offset = input_offset + mad24(mad24(h_idx, filter_w, w_idx),
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

    const int offset = input_offset + mad24(mad24(mad24(h_idx, filter_w, w_idx),
                in_channel, in_channel_idx),
            multiplier, m);
    // TODO support multiplier > 1
  }

  int2 coord = (int2)(w, h);
  WRITE_IMAGET(output, coord, values);
}

__kernel void in_out_buffer_to_image(KERNEL_ERROR_PARAMS
                                     GLOBAL_WORK_GROUP_SIZE_DIM2
                                     __global const DATA_TYPE *input, /* nhwc */
                                     __private const int input_offset,
                                     __private const int height,
                                     __private const int width,
                                     __private const int channels,
                                     __write_only image2d_t output) {
  int w = get_global_id(0);
  int h = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || h >= global_size_dim1) {
    return;
  }
#endif

  const int batch_idx = h / height;
  const int height_idx = h % height;
  const int width_idx = w % width;
  const int channel_idx = w / width * 4;
  const int offset = input_offset + ((batch_idx * height + height_idx) * width + width_idx) * channels
                           + channel_idx;

  const int size = channels - channel_idx;
  DATA_TYPE4 values = 0;
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
  WRITE_IMAGET(output, coord, values);
}

__kernel void in_out_image_to_buffer(KERNEL_ERROR_PARAMS
                                     GLOBAL_WORK_GROUP_SIZE_DIM2
                                     __global DATA_TYPE *output, /* nhwc */
                                     __private const int height,
                                     __private const int width,
                                     __private const int channels,
                                     __read_only image2d_t input) {
  int w = get_global_id(0);
  int h = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || h >= global_size_dim1) {
    return;
  }
#endif

  const int batch_idx = h / height;
  const int height_idx = h % height;
  const int width_idx = w % width;
  const int channel_idx = w / width * 4;
  const int offset = ((batch_idx * height + height_idx) * width + width_idx) * channels
                           + channel_idx;

  int2 coord = (int2)(w, h);
  DATA_TYPE4 values = READ_IMAGET(input, SAMPLER, coord);
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

__kernel void arg_buffer_to_image(KERNEL_ERROR_PARAMS
                                  GLOBAL_WORK_GROUP_SIZE_DIM2
                                  __global const DATA_TYPE *input, /* nhwc */
                                  __private const int input_offset,
                                  __private const int count,
                                  __write_only image2d_t output) {
  int w = get_global_id(0);
  int h = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || h >= global_size_dim1) {
    return;
  }
#endif

  const int offset = input_offset + w * 4;
  const int size = count - w * 4;


  DATA_TYPE4 values = 0;
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
  WRITE_IMAGET(output, coord, values);
}

__kernel void arg_image_to_buffer(KERNEL_ERROR_PARAMS
                                  GLOBAL_WORK_GROUP_SIZE_DIM2
                                  __global DATA_TYPE *output, /* nhwc */
                                  __private const int count,
                                  __read_only image2d_t input) {
  int w = get_global_id(0);
  int h = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || h >= global_size_dim1) {
    return;
  }
#endif

  const int offset = w * 4;

  int2 coord = (int2)(w, h);
  DATA_TYPE4 values = READ_IMAGET(input, SAMPLER, coord);
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


__kernel void in_out_height_buffer_to_image(KERNEL_ERROR_PARAMS
                                            GLOBAL_WORK_GROUP_SIZE_DIM2
                                            __global const DATA_TYPE *input, //nhwc
                                            __private const int input_offset,
                                            __private const int height,
                                            __private const int width,
                                            __private const int channels,
                                            __write_only image2d_t output) {
  int w = get_global_id(0);
  int h = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || h >= global_size_dim1) {
    return;
  }
#endif

  const int wc = width * channels;
  const int height_blks = (height + 3) / 4;
  const int batch_idx = h / height_blks;
  const int height_idx = (h % height_blks) << 2;
  const int width_idx = w % width;
  const int channel_idx = w / width;
  int offset = input_offset + ((batch_idx * height + height_idx) * width + width_idx) * channels
      + channel_idx;

  int size = height - height_idx;
  size = size >= 4 ? 0 : size;
  DATA_TYPE4 values = 0;
  switch(size) {
    case 0:
      values.w = *(input + offset + wc * 3);
    case 3:
      values.z = *(input + offset + wc * 2);
    case 2:
      values.y = *(input + offset + wc);
    case 1:
      values.x = *(input + offset);
  }
  int2 coord = (int2)(w, h);
  WRITE_IMAGET(output, coord, values);
}

__kernel void in_out_height_image_to_buffer(KERNEL_ERROR_PARAMS
                                            GLOBAL_WORK_GROUP_SIZE_DIM2
                                            __global DATA_TYPE *output, //nhwc
                                            __private const int height,
                                            __private const int width,
                                            __private const int channels,
                                            __read_only image2d_t input) {
  int w = get_global_id(0);
  int h = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || h >= global_size_dim1) {
    return;
  }
#endif

  const int height_blks = (height + 3) / 4;
  const int batch_idx = h / height_blks;
  const int height_idx = (h % height_blks) << 2;
  const int width_idx = w % width;
  const int channel_idx = w / width;
  int offset = ((batch_idx * height + height_idx) * width + width_idx) * channels
      + channel_idx;

  int2 coord = (int2)(w, h);
  DATA_TYPE4 values = READ_IMAGET(input, SAMPLER, coord);
  output[offset] = values.x;
  if (height_idx + 1 >= height) return;
  offset += width * channels;
  output[offset] = values.y;
  if (height_idx + 2 >= height) return;
  offset += width * channels;
  output[offset] = values.z;
  if (height_idx + 3 >= height) return;
  offset += width * channels;
  output[offset] = values.w;
}


__kernel void in_out_width_buffer_to_image(KERNEL_ERROR_PARAMS
                                           GLOBAL_WORK_GROUP_SIZE_DIM2
                                           __global const DATA_TYPE *input, /* nhwc */
                                           __private const int input_offset,
                                           __private const int height,
                                           __private const int width,
                                           __private const int channels,
                                           __write_only image2d_t output) {
  int w = get_global_id(0);
  int h = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || h >= global_size_dim1) {
    return;
  }
#endif

  const int width_blks = (width + 3) / 4;
  const int batch_idx = h / height;
  const int height_idx = h % height;
  const int width_idx = (w % width_blks) << 2;
  const int channel_idx = w / width_blks;
  const int offset = input_offset + ((batch_idx * height + height_idx) * width + width_idx) * channels
      + channel_idx;

  int size = width - width_idx;
  size = size >= 4 ? 0 : size;
  DATA_TYPE4 values = 0;
  switch(size) {
    case 0:
      values.w = *(input + offset + channels * 3);
    case 3:
      values.z = *(input + offset + channels * 2);
    case 2:
      values.y = *(input + offset + channels);
    case 1:
      values.x = *(input + offset);
  }
  int2 coord = (int2)(w, h);
  WRITE_IMAGET(output, coord, values);
}

// only support 3x3 now
__kernel void winograd_filter_buffer_to_image(KERNEL_ERROR_PARAMS
                                              GLOBAL_WORK_GROUP_SIZE_DIM2
                                              __global const DATA_TYPE *input, //Oc, Ic, H, W
                                              __private const int input_offset,
                                              __private const int in_channels,
                                              __private const int height,
                                              __private const int width,
                                              __write_only image2d_t output) {
  int w = get_global_id(0);
  int h = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || h >= global_size_dim1) {
    return;
  }
  const int out_channels = global_size_dim1;
#else
  const int out_channels = get_global_size(1);
#endif

  const int out_channel_idx = h;
  const int in_channel_idx = w << 2;
  const int offset = input_offset + (out_channel_idx * in_channels + in_channel_idx) * height * width;
  const int length = min((in_channels - in_channel_idx) * 9, 36);
  DATA_TYPE in[36] = {0};
  DATA_TYPE4 tt;
  DATA_TYPE4 tu0[4], tu1[4], tu2[4], tu3[4];

#pragma unroll
  for (short i = 0; i < length; ++i) {
    in[i] = *(input + offset + i);
  }
  tt = ((DATA_TYPE4)(in[0], in[9], in[18], in[27]) +
        (DATA_TYPE4)(in[6], in[15], in[24], in[33])) / 2;
  tu1[0] = tt + ((DATA_TYPE4)(in[3], in[12], in[21], in[30]) / 2);
  tu2[0] = tt - ((DATA_TYPE4)(in[3], in[12], in[21], in[30]) / 2);
  tt = ((DATA_TYPE4)(in[1], in[10], in[19], in[28]) +
        (DATA_TYPE4)(in[7], in[16], in[25], in[34])) / 2;
  tu1[1] = tt + ((DATA_TYPE4)(in[4], in[13], in[22], in[31]) / 2);
  tu2[1] = tt - ((DATA_TYPE4)(in[4], in[13], in[22], in[31]) / 2);
  tt = ((DATA_TYPE4)(in[2], in[11], in[20], in[29]) +
        (DATA_TYPE4)(in[8], in[17], in[26], in[35])) / 2;
  tu1[2] = tt + ((DATA_TYPE4)(in[5], in[14], in[23], in[32]) / 2);
  tu2[2] = tt - ((DATA_TYPE4)(in[5], in[14], in[23], in[32]) / 2);
  tu0[0] = (DATA_TYPE4)(in[0], in[9], in[18], in[27]);
  tu0[1] = (DATA_TYPE4)(in[1], in[10], in[19], in[28]);
  tu0[2] = (DATA_TYPE4)(in[2], in[11], in[20], in[29]);
  tu3[0] = (DATA_TYPE4)(in[6], in[15], in[24], in[33]);
  tu3[1] = (DATA_TYPE4)(in[7], in[16], in[25], in[34]);
  tu3[2] = (DATA_TYPE4)(in[8], in[17], in[26], in[35]);

  tt = (tu0[0] + tu0[2]) / 2;
  tu0[3] = tu0[2];
  tu0[2] = tt - tu0[1] / 2;
  tu0[1] = tt + tu0[1] / 2;
  tt = (tu1[0] + tu1[2]) / 2;
  tu1[3] = tu1[2];
  tu1[2] = tt - tu1[1] / 2;
  tu1[1] = tt + tu1[1] / 2;
  tt = (tu2[0] + tu2[2]) / 2;
  tu2[3] = tu2[2];
  tu2[2] = tt - tu2[1] / 2;
  tu2[1] = tt + tu2[1] / 2;
  tt = (tu3[0] + tu3[2]) / 2;
  tu3[3] = tu3[2];
  tu3[2] = tt - tu3[1] / 2;
  tu3[1] = tt + tu3[1] / 2;

  int2 coord = (int2)(w, h);

  WRITE_IMAGET(output, coord, tu0[0]);
  coord.y += out_channels;
  WRITE_IMAGET(output, coord, tu0[1]);
  coord.y += out_channels;
  WRITE_IMAGET(output, coord, tu0[2]);
  coord.y += out_channels;
  WRITE_IMAGET(output, coord, tu0[3]);
  coord.y += out_channels;

  WRITE_IMAGET(output, coord, tu1[0]);
  coord.y += out_channels;
  WRITE_IMAGET(output, coord, tu1[1]);
  coord.y += out_channels;
  WRITE_IMAGET(output, coord, tu1[2]);
  coord.y += out_channels;
  WRITE_IMAGET(output, coord, tu1[3]);
  coord.y += out_channels;

  WRITE_IMAGET(output, coord, tu2[0]);
  coord.y += out_channels;
  WRITE_IMAGET(output, coord, tu2[1]);
  coord.y += out_channels;
  WRITE_IMAGET(output, coord, tu2[2]);
  coord.y += out_channels;
  WRITE_IMAGET(output, coord, tu2[3]);
  coord.y += out_channels;

  WRITE_IMAGET(output, coord, tu3[0]);
  coord.y += out_channels;
  WRITE_IMAGET(output, coord, tu3[1]);
  coord.y += out_channels;
  WRITE_IMAGET(output, coord, tu3[2]);
  coord.y += out_channels;
  WRITE_IMAGET(output, coord, tu3[3]);
}

// only support 3x3 now
__kernel void winograd_filter_image_to_buffer(KERNEL_ERROR_PARAMS
                                              GLOBAL_WORK_GROUP_SIZE_DIM2
                                              __global DATA_TYPE *output, //Oc, Ic, H, W
                                              __private const int height,
                                              __private const int width,
                                              __private const int channel,
                                              __read_only image2d_t input) {
  const int w = get_global_id(0);
  const int h = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || h >= global_size_dim1) {
    return;
  }
#endif

  const int width_idx = w << 2;
  const int size = width - width_idx;
  int offset = h * width + width_idx;

  int2 coord = (int2)(w, h);
  DATA_TYPE4 values;
  for (short i = 0; i < 16; ++i) {
    values = READ_IMAGET(input, SAMPLER, coord);
    if (size < 4) {
      switch (size) {
        case 3:
          output[offset+2] = values.z;
        case 2:
          output[offset+1] = values.y;
        case 1:
          output[offset] = values.x;
      }
    } else {
      vstore4(values, 0, output + offset);
    }

    coord.y += height;
    offset += height * width;
  }
}
