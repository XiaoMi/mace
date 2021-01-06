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

#if CT_MODE == 0    // NONE
  const float h_in_f = h * height_scale;
  const float w_in_f = w * width_scale;
#elif CT_MODE == 1  // HALF_PIXEL
  const float h_in_f = ((float)h + 0.5f) * height_scale;
  const float w_in_f = ((float)w + 0.5f) * width_scale;
#endif

  const int h_in = min((align_corner) ? (int) round(h_in_f) :
      (int) floor(h_in_f), in_height - 1);
  const int w_in = min((align_corner) ? (int) round(w_in_f) :
      (int) floor(w_in_f), in_width - 1);

  const int in_w_offset = mul24(ch_blk, in_width);
  const int in_h_offset = mul24(b, in_height);

  const int out_w_offset = mul24(ch_blk, out_width);
  const int out_h_offset = mul24(b, out_height);

  DATA_TYPE4 out = READ_IMAGET(input, SAMPLER, (int2)(in_w_offset + w_in,
      in_h_offset + h_in));

  WRITE_IMAGET(output, (int2)(out_w_offset + w, out_h_offset + h), out);
}

