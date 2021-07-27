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


__kernel void reduce_hw(OUT_OF_RANGE_PARAMS
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
  if (bc >= global_size_dim2 || ow >= global_size_dim0)
    return;
#endif

  const int b = bc / channel_blocks;
  const int c = bc % channel_blocks;
  const int tile_w = (in_width + out_width - 1) / out_width;
  const int tile_h = (in_height + out_height - 1) / out_height;
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

  int out_x = mad24(c, out_width, ow);
  int out_y = mad24(b, out_height, oh);
  WRITE_IMAGET(output, (int2)(out_x, out_y), out);
}

__kernel void reduce_c(OUT_OF_RANGE_PARAMS
                       GLOBAL_WORK_GROUP_SIZE_DIM3
                       __read_only image2d_t input,
                       __private const int height,
                       __private const int width,
                       __private const int channels,
                       __private const int channel_blocks,
                       __private const int in_ch_blks,
                       __write_only image2d_t output) {
  const int out_ch_blks = global_size_dim0;
  const int oc = get_global_id(0);
  const int w = get_global_id(1);
  const int bh = get_global_id(2);
#ifndef NON_UNIFORM_WORK_GROUP
  if (bh >= global_size_dim2 || w >= global_size_dim1 || oc >= global_size_dim0)
    return;
#endif
  const int b = bh / height;
  const int h = bh - b * height;
  const int tile_ch_blk = (in_ch_blks + out_ch_blks -1) / out_ch_blks;
  const int start_ch_blk = tile_ch_blk * oc;
  const int size_ch_blk = select(tile_ch_blk, in_ch_blks - start_ch_blk, oc >= out_ch_blks -1);
  const int end_ch_blk = start_ch_blk + size_ch_blk;
  DATA_TYPE4 in;
  DATA_TYPE4 out = INIT_REDUCE_VALUE;
  for (int ch = start_ch_blk; ch < end_ch_blk; ++ch) {
    int pos_x = mad24(ch, width, w);
    int pos_y = mad24(b, height, h);
    in = READ_IMAGET(input, SAMPLER, (int2)(pos_x, pos_y));
#if defined(NOT_DIVISIBLE_FOUR) && \
  ((REDUCE_TYPE == 1) || (REDUCE_TYPE == 2) || (REDUCE_TYPE == 3))

#if REDUCE_TYPE == 1 // MIN
#define SCALAR_INIT_REDUCE_VALUE MAXFLOAT
#elif REDUCE_TYPE == 2  // MAX
#define SCALAR_INIT_REDUCE_VALUE -MAXFLOAT
#elif REDUCE_TYPE == 3  // PROD
#define SCALAR_INIT_REDUCE_VALUE 1
#endif
    if (ch == channel_blocks -1) {
      int blank_channels = mad24(4, channel_blocks, -channels);
      switch (blank_channels) {
        case 3:
          in.y = SCALAR_INIT_REDUCE_VALUE;
        case 2:
          in.z = SCALAR_INIT_REDUCE_VALUE;
        case 1:
          in.w = SCALAR_INIT_REDUCE_VALUE;
      }
    }
#endif
    out = REDUCE_VALUE(out, in);
  }
  // Reduce vector to scalar
  if (out_ch_blks == 1) {
    out.x = REDUCE_VALUE(out.x, out.y);
    out.z = REDUCE_VALUE(out.z, out.w);
    out.x = REDUCE_VALUE(out.x, out.z);
#if REDUCE_TYPE == 0
    out.x = out.x / channels;
#endif
    out.y = 0;
    out.z = 0;
    out.w = 0;
  }
  int out_x = mad24(oc, width, w);
  int out_y = mad24(b, height, h);
  WRITE_IMAGET(output, (int2)(out_x, out_y), out);
}
