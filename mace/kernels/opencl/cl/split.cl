#include <common.h>

__kernel void split(KERNEL_ERROR_PARAMS
                    GLOBAL_WORK_GROUP_SIZE_DIM3
                    __read_only image2d_t input,
                    __private const int chan_blk_offset,
                    __write_only image2d_t output) {
  const int chan_blk_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (chan_blk_idx >= global_size_dim0 || width_idx >= global_size_dim1
      || hb_idx >= global_size_dim2) {
    return;
  }
#endif
  const int width = global_size_dim1;

  DATA_TYPE4 data = READ_IMAGET(input, SAMPLER,
                                (int2)(mad24(chan_blk_idx + chan_blk_offset,
                                             width, width_idx), hb_idx));

  const int pos = mad24(chan_blk_idx, width, width_idx);
  WRITE_IMAGET(output, (int2)(pos, hb_idx), data);
}
