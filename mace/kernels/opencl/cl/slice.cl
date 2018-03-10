#include <common.h>

__kernel void slice(__read_only image2d_t input,
                    __private const int chan_blk_offset,
                    __write_only image2d_t output) {
  const int chan_blk_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int width = get_global_size(1);
  const int hb_idx = get_global_id(2);
  DATA_TYPE4 data = READ_IMAGET(input, SAMPLER,
                                (int2)(mad24(chan_blk_idx + chan_blk_offset,
                                             width, width_idx), hb_idx));
  WRITE_IMAGET(output,
               (int2)(mad24(chan_blk_idx, width, width_idx), hb_idx), data);
}
