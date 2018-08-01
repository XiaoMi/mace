#include <common.h>

__kernel void resize_bilinear_nocache(KERNEL_ERROR_PARAMS
                                      GLOBAL_WORK_GROUP_SIZE_DIM3
                                      __read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
                                      __write_only image2d_t output,
                                      __private const float height_scale,
                                      __private const float width_scale,
                                      __private const int in_height,
                                      __private const int in_width,
                                      __private const int out_height) {

  const int ch_blk = get_global_id(0);
  const int w = get_global_id(1);
  const int hb = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (ch_blk >= global_size_dim0 || w >= global_size_dim1
      || hb >= global_size_dim2) {
    return;
  }
#endif
  const int ch_blks = global_size_dim0;
  const int out_width = global_size_dim1;

  const int b = hb / out_height;
  const int h = hb % out_height;

  const float h_in = h * height_scale;
  const float w_in = w * width_scale;
  const int h_lower = max(0, (int) floor(h_in));
  const int h_upper = min(in_height - 1, h_lower + 1);
  const int w_lower = max(0, (int) floor(w_in));
  const int w_upper = min(in_width - 1, w_lower + 1);

  const float h_lerp = h_in - h_lower;
  const float w_lerp = w_in - w_lower;

  const int in_w_offset = mul24(ch_blk, in_width);
  const int in_h_offset = mul24(b, in_height);

  DATA_TYPE4 top_left = READ_IMAGET(input, SAMPLER,
          (int2)(in_w_offset + w_lower, in_h_offset + h_lower));
  DATA_TYPE4 top_right = READ_IMAGET(input, SAMPLER,
          (int2)(in_w_offset + w_upper, in_h_offset + h_lower));
  DATA_TYPE4 bottom_left = READ_IMAGET(input, SAMPLER,
          (int2)(in_w_offset + w_lower, in_h_offset + h_upper));
  DATA_TYPE4 bottom_right = READ_IMAGET(input, SAMPLER,
          (int2)(in_w_offset + w_upper, in_h_offset + h_upper));

  DATA_TYPE4 top = mad((top_right - top_left), w_lerp, top_left);
  DATA_TYPE4 bottom = mad((bottom_right - bottom_left), w_lerp, bottom_left);
  DATA_TYPE4 out = mad((bottom - top), h_lerp, top);

  const int out_w_offset = mul24(ch_blk, out_width);
  const int out_h_offset = mul24(b, out_height);

  WRITE_IMAGET(output, (int2)(out_w_offset + w, out_h_offset + h), out);
}

