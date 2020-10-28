#include <common.h>

#if REDUCE_TYPE == 1
#define INIT_REDUCE_VALUE (DATA_TYPE4){MAXFLOAT, MAXFLOAT, MAXFLOAT, MAXFLOAT}
#define REDUCE_VALUE(x, y) fmin(x, y)
#elif REDUCE_TYPE == 2  // MAX
#define INIT_REDUCE_VALUE (DATA_TYPE4){-MAXFLOAT, -MAXFLOAT, -MAXFLOAT, -MAXFLOAT}
#define REDUCE_VALUE(x, y) fmax(x, y)
#elif REDUCE_TYPE == 3  // PROD
#define INIT_REDUCE_VALUE (DATA_TYPE4){1, 1, 1, 1}
#define REDUCE_VALUE(x, y) (x * y)
#else  // MEAN or SUM
#define INIT_REDUCE_VALUE (DATA_TYPE4){0, 0, 0, 0}
#define REDUCE_VALUE(x, y) (x + y)
#endif


__kernel void reduce(OUT_OF_RANGE_PARAMS
                     GLOBAL_WORK_GROUP_SIZE_DIM3
                     __read_only image2d_t input,
                     __private const int out_height,
                     __private const int out_width,
                     __private const int in_height,
                     __private const int in_width,
                     __private const int org_height,
                     __private const int org_width,
                     __private const int channel_blocks,
                     __write_only image2d_t output) {
  const int ow = get_global_id(0);
  const int oh = get_global_id(1);
  const int bc = get_global_id(2);
#ifndef NON_UNIFORM_WORK_GROUP
  if (bc >= global_size_dim2)
    return;
#endif

  const int b = bc / channel_blocks;
  const int c = bc % channel_blocks;
  const int tile_w = in_width / out_width;
  const int tile_h = in_height / out_height;
  const int start_w = tile_w * ow;
  const int start_h = tile_h * oh;

  const int size_w = select(tile_w, in_width - start_w, ow >= out_width - 1);
  const int size_h = select(tile_h, in_height - start_h, oh >= out_height - 1);
  const int end_h = start_h + size_h;
  const int end_w = start_w + size_w;

  DATA_TYPE4 in;
  DATA_TYPE4 out = INIT_REDUCE_VALUE;
#pragma unroll
  for (int h = start_h; h < end_h; ++h) {
    for (int w = start_w; w < end_w; ++w) {
      int pos_x = mad24(c, in_width, w);
      int pos_y = mad24(b, in_height, h);
      in = READ_IMAGET(input, SAMPLER, (int2)(pos_x, pos_y));
      out = REDUCE_VALUE(out, in);
    }
  }
#if REDUCE_TYPE == 0
  if (out_height == 1 && out_width == 1) {
    out = out / (org_height * org_width);
  }
#endif

  int pos_x = mad24(c, out_width, ow);
  int pos_y = mad24(b, out_height, oh);
  WRITE_IMAGET(output, (int2)(pos_x, pos_y), out);
}
