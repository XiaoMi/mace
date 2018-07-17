#include <common.h>

__kernel void crop(KERNEL_ERROR_PARAMS
                   GLOBAL_WORK_GROUP_SIZE_DIM3
                   __read_only image2d_t input,
                   __private const int offset_b,
                   __private const int offset_h,
                   __private const int offset_w,
                   __private const int offset_chan_blk,
                   __private const int in_height,
                   __private const int in_width,
                   __private const int out_height,
                   __private const int out_width,
                   __write_only image2d_t output) {
  const int chan_blk_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (chan_blk_idx >= global_size_dim0 || width_idx >= global_size_dim1
      || hb_idx >= global_size_dim2) {
    return;
  }
  const int width = global_size_dim1;
#else
  const int width = get_global_size(1);
#endif

  const int b = hb_idx / out_height;
  const int h = hb_idx % out_height;
  const int in_chan_blk_idx = chan_blk_idx + offset_chan_blk;
  const int in_width_idx = width_idx + offset_w;
  const int in_h = h + offset_h;
  const int in_b = b + offset_b;
  const int in_hb_idx = mad24(in_b, in_height, in_h);
  const int in_pos = mad24(in_chan_blk_idx, in_width, in_width_idx);

  DATA_TYPE4 data = READ_IMAGET(input, SAMPLER,
                                (int2)(in_pos, in_hb_idx));

  const int pos = mad24(chan_blk_idx, width, width_idx);
  WRITE_IMAGET(output, (int2)(pos, hb_idx), data);
}
