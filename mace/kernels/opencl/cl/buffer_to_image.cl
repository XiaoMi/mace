#include <common.h>

__kernel void filter_buffer_to_image(KERNEL_ERROR_PARAMS
                                     GLOBAL_WORK_GROUP_SIZE_DIM2
                                     __global const DATA_TYPE *input, /* OIHW */
                                     __private const int input_offset,
                                     __private const int out_channel,
                                     __private const int filter_h,
                                     __private const int filter_w,
                                     __private const int inner_size,
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
  const int offset = input_offset +
      mad24(out_channel_idx, inner_size,
          mad24(mad24(in_channel_idx, filter_h, h_idx), filter_w, w_idx));

  DATA_TYPE4 values = 0;
  if (out_channel_idx < out_channel) {
    const int size = out_channel - out_channel_idx;
    if (size < 4) {
      switch (size) {
        case 3:
          values.z = *(input + offset + 2 * inner_size);
        case 2:
          values.y = *(input + offset + 1 * inner_size);
        case 1:
          values.x = *(input + offset);
      }
    } else {
      values.w = *(input + offset + 3 * inner_size);
      values.z = *(input + offset + 2 * inner_size);
      values.y = *(input + offset + 1 * inner_size);
      values.x = *(input + offset);
    }
  }

  int2 coord = (int2)(w, h);
  WRITE_IMAGET(output, coord, values);
}

__kernel void filter_image_to_buffer(KERNEL_ERROR_PARAMS
                                     GLOBAL_WORK_GROUP_SIZE_DIM2
                                     __global DATA_TYPE *output, /* OIHW */
                                     __private const int out_channel,
                                     __private const int filter_h,
                                     __private const int filter_w,
                                     __private const int inner_size,
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
  const int offset =
      mad24(out_channel_idx, inner_size,
            mad24(mad24(in_channel_idx, filter_h, h_idx), filter_w, w_idx));

  if (out_channel_idx < out_channel) {
    int2 coord = (int2)(w, h);
    DATA_TYPE4 values = READ_IMAGET(input, SAMPLER, coord);
    const int size = (out_channel - out_channel_idx);
    if (size < 4) {
      switch (size) {
        case 3:
          output[offset + 2 * inner_size] = values.z;
        case 2:
          output[offset + 1 * inner_size] = values.y;
        case 1:
          output[offset] = values.x;
      }
    } else {
      output[offset + 3 * inner_size] = values.w;
      output[offset + 2 * inner_size] = values.z;
      output[offset + 1 * inner_size] = values.y;
      output[offset] = values.x;
    }
  }
}

// TODO(liuqi): Support multiplier > 1
__kernel void dw_filter_buffer_to_image(KERNEL_ERROR_PARAMS
                                        GLOBAL_WORK_GROUP_SIZE_DIM2
                                        __global const DATA_TYPE *input, /* MIHW */
                                        __private const int input_offset,
                                        __private const int multiplier,
                                        __private const int in_channel,
                                        __private const int filter_h,
                                        __private const int filter_w,
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

    const int offset = input_offset
        + mad24(mad24(in_channel_idx, filter_h, h_idx), filter_w, w_idx);

    const int hw_size = mul24(filter_h, filter_w);
    const int size = in_channel - in_channel_idx;
    if (in_channel_idx < in_channel) {
      if (size < 4) {
        switch(size) {
          case 3:
            values.z = *(input + offset + 2 * hw_size);
          case 2:
            values.y = *(input + offset + 1 * hw_size);
          case 1:
            values.x = *(input + offset);
        }
      } else {
        values.x = *(input + offset);
        values.y = *(input + offset + 1 * hw_size);
        values.z = *(input + offset + 2 * hw_size);
        values.w = *(input + offset + 3 * hw_size);
      }
    }
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
                                  __global const DATA_TYPE *input,
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
                                  __global DATA_TYPE *output,
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
  const int offset = input_offset
      + ((batch_idx * height + height_idx) * width + width_idx) * channels
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

__kernel void weight_height_buffer_to_image(KERNEL_ERROR_PARAMS
                                            GLOBAL_WORK_GROUP_SIZE_DIM2
                                            __global const DATA_TYPE *input, // OIHW
                                            __private const int input_offset,
                                            __private const int out_channels,
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
#endif
  const int inner_size = global_size_dim0;

  const int out_chan_idx = h << 2;
  const int in_chan_idx = w % in_channels;
  const int hw_idx = w / in_channels;
  const int height_idx = hw_idx / width;
  const int width_idx = hw_idx % width;
  int offset = input_offset +
      mad24(out_chan_idx, inner_size,
            mad24(mad24(in_chan_idx, height, height_idx), width, width_idx));

  int size = out_channels - out_chan_idx;
  size = size >= 4 ? 0 : size;
  DATA_TYPE4 values = 0;
  switch (size) {
    case 0:
      values.w = *(input + offset + inner_size * 3);
    case 3:
      values.z = *(input + offset + inner_size * 2);
    case 2:
      values.y = *(input + offset + inner_size);
    case 1:
      values.x = *(input + offset);
  }
  int2 coord = (int2)(w, h);
  WRITE_IMAGET(output, coord, values);
}

__kernel void weight_height_image_to_buffer(KERNEL_ERROR_PARAMS
                                            GLOBAL_WORK_GROUP_SIZE_DIM2
                                            __global DATA_TYPE *output, //OIHW
                                            __private const int out_channels,
                                            __private const int in_channels,
                                            __private const int height,
                                            __private const int width,
                                            __read_only image2d_t input) {
  int w = get_global_id(0);
  int h = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || h >= global_size_dim1) {
    return;
  }
#endif
  const int inner_size = global_size_dim0;

  const int out_chan_idx = h << 2;
  const int in_chan_idx = w % in_channels;
  const int hw_idx = w / in_channels;
  const int height_idx = hw_idx / width;
  const int width_idx = hw_idx % width;
  int offset =
      mad24(out_chan_idx, inner_size,
            mad24(mad24(in_chan_idx, height, height_idx), width, width_idx));

  int2 coord = (int2)(w, h);
  DATA_TYPE4 values = READ_IMAGET(input, SAMPLER, coord);
  output[offset] = values.x;
  if (out_chan_idx + 1 >= out_channels) return;
  offset += inner_size;
  output[offset] = values.y;
  if (out_chan_idx + 2 >= out_channels) return;
  offset += inner_size;
  output[offset] = values.z;
  if (out_chan_idx + 3 >= out_channels) return;
  offset += inner_size;
  output[offset] = values.w;
}


__kernel void weight_width_buffer_to_image(KERNEL_ERROR_PARAMS
                                           GLOBAL_WORK_GROUP_SIZE_DIM2
                                           __global const DATA_TYPE *input, // OIHW
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
#endif
  const int out_channels = global_size_dim1;
  const int in_chan_blks = (in_channels + 3) >> 2;
  const int hw_size = height * width;
  const int inner_size = in_channels * hw_size;

  const int out_chan_idx = h;
  const int in_chan_idx = (w % in_chan_blks) << 2;
  const int hw_idx = w / in_chan_blks;
  const int height_idx = hw_idx / width;
  const int width_idx = hw_idx % width;
  int offset = input_offset +
      mad24(out_chan_idx, inner_size,
            mad24(mad24(in_chan_idx, height, height_idx), width, width_idx));


  int size = in_channels - in_chan_idx;
  size = size >= 4 ? 0 : size;
  DATA_TYPE4 values = 0;
  switch (size) {
    case 0:
      values.w = *(input + offset + hw_size * 3);
    case 3:
      values.z = *(input + offset + hw_size * 2);
    case 2:
      values.y = *(input + offset + hw_size);
    case 1:
      values.x = *(input + offset);
  }
  int2 coord = (int2)(w, h);
  WRITE_IMAGET(output, coord, values);
}

__kernel void weight_width_image_to_buffer(KERNEL_ERROR_PARAMS
                                           GLOBAL_WORK_GROUP_SIZE_DIM2
                                           __global DATA_TYPE *output, // OIHW
                                           __private const int in_channels,
                                           __private const int height,
                                           __private const int width,
                                           __read_only image2d_t input) {
  int w = get_global_id(0);
  int h = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || h >= global_size_dim1) {
    return;
  }
#endif
  const int out_channels = global_size_dim1;
  const int in_chan_blks = (in_channels + 3) >> 2;
  const int hw_size = height * width;
  const int inner_size = in_channels * hw_size;

  const int out_chan_idx = h;
  const int in_chan_idx = (w % in_chan_blks) << 2;
  const int hw_idx = w / in_chan_blks;
  const int height_idx = hw_idx / width;
  const int width_idx = hw_idx % width;
  int offset =
      mad24(out_chan_idx, inner_size,
            mad24(mad24(in_chan_idx, height, height_idx), width, width_idx));

  int2 coord = (int2)(w, h);
  DATA_TYPE4 values = READ_IMAGET(input, SAMPLER, coord);
  output[offset] = values.x;
  if (in_chan_idx + 1 >= in_channels) return;
  offset += hw_size;
  output[offset] = values.y;
  if (in_chan_idx + 2 >= in_channels) return;
  offset += hw_size;
  output[offset] = values.z;
  if (in_chan_idx + 3 >= in_channels) return;
  offset += hw_size;
  output[offset] = values.w;
}

// only support 3x3 now
__kernel void winograd_filter_buffer_to_image_2x2(KERNEL_ERROR_PARAMS
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
#endif
  const int out_channels = global_size_dim1;

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
__kernel void winograd_filter_image_to_buffer_2x2(KERNEL_ERROR_PARAMS
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

// only support 3x3 now
__kernel void winograd_filter_buffer_to_image_6x6(KERNEL_ERROR_PARAMS
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
#endif
  const int out_channels = global_size_dim1;

  const int out_channel_idx = h;
  const int in_channel_idx = w << 2;
  const int offset = input_offset + (out_channel_idx * in_channels + in_channel_idx) * height * width;
  const int length = min((in_channels - in_channel_idx) * 9, 36);
  DATA_TYPE in[36] = {0};
  DATA_TYPE4 tt0, tt1, t1;
  DATA_TYPE4 tu0[3], tu1[3], tu2[3], tu3[3], tu4[3], tu5[3], tu6[3], tu7[3];

  const float a = -0.222222222f;
  const float b = 0.011111111f;
  const float c = 0.005555556f;

#pragma unroll
  for (short i = 0; i < length; ++i) {
    in[i] = *(input + offset + i);
  }

  tu0[0] = (DATA_TYPE4)(in[0], in[9], in[18], in[27]);
  t1 = (DATA_TYPE4)(in[3], in[12], in[21], in[30]);
  tu7[0] = (DATA_TYPE4)(in[6], in[15], in[24], in[33]);

  tt0 = tu0[0] + tu7[0];
  tt1 = t1;
  tu1[0] = mad(tt0 + tt1, a, 0);
  tu2[0] = mad(tt0 - tt1, a, 0);
  tt0 = mad(tu7[0], 4, tu0[0]);
  tt1 = mad(t1, 2, 0);
  tu3[0] = mad(tt0 + tt1, b, 0);
  tu4[0] = mad(tt0 - tt1, b, 0);
  tt0 = mad(tu0[0], 4, tu7[0]);
  tt1 = mad(t1, 2, 0);
  tu5[0] = mad(tt0 + tt1, c, 0);
  tu6[0] = mad(tt0 - tt1, c, 0);

  tu0[1] = (DATA_TYPE4)(in[1], in[10], in[19], in[28]);
  t1 = (DATA_TYPE4)(in[4], in[13], in[22], in[31]);
  tu7[1] = (DATA_TYPE4)(in[7], in[16], in[25], in[34]);

  tt0 = tu0[1] + tu7[1];
  tt1 = t1;
  tu1[1] = mad(tt0 + tt1, a, 0);
  tu2[1] = mad(tt0 - tt1, a, 0);

  tt0 = mad(tu7[1], 4, tu0[1]);
  tt1 = mad(t1, 2, 0);
  tu3[1] = mad(tt0 + tt1, b, 0);
  tu4[1] = mad(tt0 - tt1, b, 0);

  tt0 = mad(tu0[1], 4, tu7[1]);
  tt1 = mad(t1, 2, 0);
  tu5[1] = mad(tt0 + tt1, c, 0);
  tu6[1] = mad(tt0 - tt1, c, 0);

  tu0[2] = (DATA_TYPE4)(in[2], in[11], in[20], in[29]);
  t1 = (DATA_TYPE4)(in[5], in[14], in[23], in[32]);
  tu7[2] = (DATA_TYPE4)(in[8], in[17], in[26], in[35]);

  tt0 = tu0[2] + tu7[2];
  tt1 = t1;
  tu1[2] = mad(tt0 + tt1, a, 0);
  tu2[2] = mad(tt0 - tt1, a, 0);

  tt0 = mad(tu7[2], 4, tu0[2]);
  tt1 = mad(t1, 2, 0);
  tu3[2] = mad(tt0 + tt1, b, 0);
  tu4[2] = mad(tt0 - tt1, b, 0);

  tt0 = mad(tu0[2], 4, tu7[2]);
  tt1 = mad(t1, 2, 0);
  tu5[2] = mad(tt0 + tt1, c, 0);
  tu6[2] = mad(tt0 - tt1, c, 0);

#define PROCESS(i)                             \
  t1 = tu##i[0];                               \
  WRITE_IMAGET(output, (int2)(w, h), t1);      \
  h += out_channels;                           \
  tt0 = tu##i[0] + tu##i[2];                   \
  tt1 = tu##i[1];                              \
  t1 = mad(tt0 + tt1, a, 0);                   \
  WRITE_IMAGET(output, (int2)(w, h), t1);      \
  h += out_channels;                           \
  t1 = mad(tt0 - tt1, a, 0);                   \
  WRITE_IMAGET(output, (int2)(w, h), t1);      \
  h += out_channels;                           \
  tt0 = mad(tu##i[2], 4, tu##i[0]);            \
  tt1 = mad(tu##i[1], 2, 0);                   \
  t1 = mad(tt0 + tt1, b, 0);                   \
  WRITE_IMAGET(output, (int2)(w, h), t1);      \
  h += out_channels;                           \
  t1 = mad(tt0 - tt1, b, 0);                   \
  WRITE_IMAGET(output, (int2)(w, h), t1);      \
  h += out_channels;                           \
  tt0 = mad(tu##i[0], 4, tu##i[2]);            \
  tt1 = mad(tu##i[1], 2, 0);                   \
  t1 = mad(tt0 + tt1, c, 0);                   \
  WRITE_IMAGET(output, (int2)(w, h), t1);      \
  h += out_channels;                           \
  t1 = mad(tt0 - tt1, c, 0);                   \
  WRITE_IMAGET(output, (int2)(w, h), t1);      \
  h += out_channels;                           \
  t1 = tu##i[2];                               \
  WRITE_IMAGET(output, (int2)(w, h), t1);      \
  h += out_channels;                           \

PROCESS(0);
PROCESS(1);
PROCESS(2);
PROCESS(3);
PROCESS(4);
PROCESS(5);
PROCESS(6);
PROCESS(7);

#undef PROCESS

}
__kernel void winograd_filter_image_to_buffer_6x6(KERNEL_ERROR_PARAMS
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
  for (short i = 0; i < 64; ++i) {
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

// only support 3x3 now
__kernel void winograd_filter_buffer_to_image_4x4(KERNEL_ERROR_PARAMS
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
#endif
  const int out_channels = global_size_dim1;

  const int out_channel_idx = h;
  const int in_channel_idx = w << 2;
  const int offset = input_offset + (out_channel_idx * in_channels + in_channel_idx) * height * width;
  const int length = min((in_channels - in_channel_idx) * 9, 36);
  DATA_TYPE in[36] = {0};
  DATA_TYPE4 tt0, tt1, tt2;
  DATA_TYPE4 tu0[3], tu1[3], tu2[3], tu3[3], tu4[3], tu5[3];
  const float a = 0.25f;
  const float b = -0.166666667f;
  const float c = 0.041666667f;

#pragma unroll
  for (short i = 0; i < length; ++i) {
    in[i] = *(input + offset + i);
  }

  tt0 = (DATA_TYPE4)(in[0], in[9], in[18], in[27]);
  tt1 = (DATA_TYPE4)(in[3], in[12], in[21], in[30]);
  tt2 = (DATA_TYPE4)(in[6], in[15], in[24], in[33]);

  tu0[0] = mad(tt0, a, 0);
  tu1[0] = mad((tt0 + tt1 + tt2), b, 0);
  tu2[0] = mad((tt0 - tt1 + tt2), b, 0);
  tt0 = mad(tt2, 4, tt0);
  tu3[0] = mad(mad(tt1, 2, tt0), c, 0);
  tu4[0] = mad(mad(tt1, -2, tt0), c, 0);

  tu5[0] = tt2;

  tt0 = (DATA_TYPE4)(in[1], in[10], in[19], in[28]);
  tt1 = (DATA_TYPE4)(in[4], in[13], in[22], in[31]);
  tt2 = (DATA_TYPE4)(in[7], in[16], in[25], in[34]);

  tu0[1] = mad(tt0, a, 0);
  tu1[1] = mad((tt0 + tt1 + tt2), b, 0);
  tu2[1] = mad((tt0 - tt1 + tt2), b, 0);
  tt0 = mad(tt2, 4, tt0);
  tu3[1] = mad(mad(tt1, 2, tt0), c, 0);
  tu4[1] = mad(mad(tt1, -2, tt0), c, 0);

  tu5[1] = tt2;

  tt0 = (DATA_TYPE4)(in[2], in[11], in[20], in[29]);
  tt1 = (DATA_TYPE4)(in[5], in[14], in[23], in[32]);
  tt2 = (DATA_TYPE4)(in[8], in[17], in[26], in[35]);

  tu0[2] = mad(tt0, a, 0);
  tu1[2] = mad((tt0 + tt1 + tt2), b, 0);
  tu2[2] = mad((tt0 - tt1 + tt2), b, 0);
  tt0 = mad(tt2, 4, tt0);
  tu3[2] = mad(mad(tt1, 2, tt0), c, 0);
  tu4[2] = mad(mad(tt1, -2, tt0), c, 0);

  tu5[2] = tt2;

#define PROCESS(i)                               \
    tt2 = mad(tu##i[0], a, 0);                   \
    WRITE_IMAGET(output, (int2)(w, h), tt2);     \
    h += out_channels;                           \
    tt0 = tu##i[1];                              \
    tt1 = tu##i[0] + tu##i[2];                   \
    tt2 = mad((tt0 + tt1), b, 0);                \
    WRITE_IMAGET(output, (int2)(w, h), tt2);     \
    h += out_channels;                           \
    tt2 = mad(tt1 - tt0, b, 0);                  \
    WRITE_IMAGET(output, (int2)(w, h), tt2);     \
    h += out_channels;                           \
    tt0 = mad(tu##i[2], 4, tu##i[0]);            \
    tt1 = 2 * tu##i[1];                          \
    tt2 = mad(tt0 + tt1, c, 0);                  \
    WRITE_IMAGET(output, (int2)(w, h), tt2);     \
    h += out_channels;                           \
    tt2 = mad(tt0 - tt1, c, 0);                  \
    WRITE_IMAGET(output, (int2)(w, h), tt2);     \
    h += out_channels;                           \
    tt2 = tu##i[2];                              \
    WRITE_IMAGET(output, (int2)(w, h), tt2);     \
    h += out_channels;                           \

  PROCESS(0);
  PROCESS(1);
  PROCESS(2);
  PROCESS(3);
  PROCESS(4);
  PROCESS(5);

#undef PROCESS

}
__kernel void winograd_filter_image_to_buffer_4x4(KERNEL_ERROR_PARAMS
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
  for (short i = 0; i < 36; ++i) {
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