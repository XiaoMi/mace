#include <common.h>

__kernel void resize_nearest_neighbor_nocache(
    OUT_OF_RANGE_PARAMS
    GLOBAL_WORK_GROUP_SIZE_DIM3
    __read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
    __write_only image2d_t output,
    __private const float height_scale,
    __private const float width_scale,
    __private const int in_height,
    __private const int in_width,
    __private const int out_height,
    __private const int align_corner) {
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
  const int h = hb - mul24(b, out_height);

  const int h_in = min((align_corner) ? (int) round(h * height_scale) :
      (int) floor(h * height_scale), in_height - 1);
  const int w_in = min((align_corner) ? (int) round(w * width_scale) :
      (int) floor(w * width_scale), in_width - 1);

  const int in_w_offset = mul24(ch_blk, in_width);
  const int in_h_offset = mul24(b, in_height);

  const int out_w_offset = mul24(ch_blk, out_width);
  const int out_h_offset = mul24(b, out_height);

  DATA_TYPE4 out = READ_IMAGET(input, SAMPLER, (int2)(in_w_offset + w_in,
      in_h_offset + h_in));

  WRITE_IMAGET(output, (int2)(out_w_offset + w, out_h_offset + h), out);
}

