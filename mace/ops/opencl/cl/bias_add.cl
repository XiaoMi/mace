#include <common.h>
// Supported data types: half/float
__kernel void bias_add(OUT_OF_RANGE_PARAMS
                       GLOBAL_WORK_GROUP_SIZE_DIM3
                       __private const int input_height,
                       __read_only image2d_t input,
                       __read_only image2d_t bias,
                       __write_only image2d_t output) {
  const int ch_blk = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (ch_blk >= global_size_dim0 || width_idx >= global_size_dim1
      || hb_idx >= global_size_dim2) {
    return;
  }
#endif
  const int width = global_size_dim1;

  const int pos = mad24(ch_blk, width, width_idx);
  DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
  const int b_idx = select(0, hb_idx / input_height, input_height > 0);
  DATA_TYPE4 bias_value = READ_IMAGET(bias, SAMPLER, (int2)(ch_blk, b_idx));
  DATA_TYPE4 out = in + bias_value;

  WRITE_IMAGET(output, (int2)(pos, hb_idx), out);
}
