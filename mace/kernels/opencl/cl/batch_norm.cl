#include <common.h>
// Supported data types: half/float
__kernel void batch_norm(__read_only image2d_t input,
                         __read_only image2d_t scale,
                         __read_only image2d_t offset,
                         __read_only image2d_t mean,
                         __read_only image2d_t var,
                         __private const DATA_TYPE epsilon,
                         __write_only image2d_t output) {
  const int ch_blk = get_global_id(0);
  const int w = get_global_id(1);
  const int hb = get_global_id(2);
  const int width = get_global_size(1);

  DATA_TYPE4 scale_value = READ_IMAGET(scale, SAMPLER, (int2)(ch_blk, 0));
  DATA_TYPE4 offset_value = READ_IMAGET(offset, SAMPLER, (int2)(ch_blk, 0));
  DATA_TYPE4 mean_value = READ_IMAGET(mean, SAMPLER, (int2)(ch_blk, 0));
  DATA_TYPE4 var_value = READ_IMAGET(var, SAMPLER, (int2)(ch_blk, 0));

  DATA_TYPE4 new_scale = scale_value * rsqrt(var_value + (DATA_TYPE4)epsilon);
  DATA_TYPE4 new_offset = offset_value - mean_value * new_scale;

  const int pos = ch_blk * width + w;

  DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(pos, hb));
  DATA_TYPE4 out = in * new_scale + new_offset;
  WRITE_IMAGET(output, (int2)(pos, hb), out);
}
