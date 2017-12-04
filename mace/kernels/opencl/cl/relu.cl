#include <common.h>

// Supported data type: half/float
__kernel void relu(__read_only image2d_t input,
                   __write_only image2d_t output) {
  const int ch_blk = get_global_id(0);
  const int w = get_global_id(1);
  const int hb = get_global_id(2);
  const int width = get_global_size(1);

  const int pos = ch_blk * width + w;
  DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(pos, hb));
  DATA_TYPE4 out = fmax(in, 0);
  WRITE_IMAGET(output, (int2)(pos, hb), out);
}

__kernel void relux(__read_only image2d_t input,
                    __private const DATA_TYPE max_limit,
                    __write_only image2d_t output) {
  const int ch_blk = get_global_id(0);
  const int w = get_global_id(1);
  const int hb = get_global_id(2);
  const int width = get_global_size(1);

  const int pos = ch_blk * width + w;
  DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(pos, hb));
  DATA_TYPE4 out = clamp(in, 0, max_limit);
  WRITE_IMAGET(output, (int2)(pos, hb), out);
}
