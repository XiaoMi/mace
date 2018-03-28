#include <common.h>

__kernel void slice(__read_only image2d_t input,
                    __private const int chan_blk_offset,
#ifndef USE_QUALCOMM_OPENCL_2_0
                    __write_only image2d_t output,
                    __private const int global_size_dim0,
                    __private const int global_size_dim1,
                    __private const int global_size_dim2) {
#else
                    __write_only image2d_t output) {
#endif

  const int chan_blk_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);

#ifndef USE_QUALCOMM_OPENCL_2_0
  if (chan_blk_idx >= global_size_dim0 || width_idx >= global_size_dim1
      || hb_idx >= global_size_dim2) {
    return;
  }
  const int width = global_size_dim1;
#else
  const int width = get_global_size(1);
#endif

  DATA_TYPE4 data = READ_IMAGET(input, SAMPLER,
                                (int2)(mad24(chan_blk_idx + chan_blk_offset,
                                             width, width_idx), hb_idx));
  WRITE_IMAGET(output,
               (int2)(mad24(chan_blk_idx, width, width_idx), hb_idx), data);
}
