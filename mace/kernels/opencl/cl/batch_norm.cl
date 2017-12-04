#include <common.h>
// Supported data types: half/float
__kernel void batch_norm(__read_only image2d_t input,
                         __read_only image2d_t scale,
                         __read_only image2d_t offset,
                         __read_only image2d_t mean,
                         __read_only image2d_t var,
                         __global const DATA_TYPE *epsilon,
                         __write_only image2d_t output) {
  const int ch_blk = get_global_id(0);
  const int w_blk = get_global_id(1);
  const int hb_blk = get_global_id(2);
  const int width = get_global_size(1);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;


  DATA_TYPE4 scale_value = READ_IMAGET(scale, sampler, (int2)(ch_blk, 0));
  DATA_TYPE4 offset_value = READ_IMAGET(offset, sampler, (int2)(ch_blk, 0));
  DATA_TYPE4 mean_value = READ_IMAGET(mean, sampler, (int2)(ch_blk, 0));
  DATA_TYPE4 var_value = READ_IMAGET(var, sampler, (int2)(ch_blk, 0));

  DATA_TYPE4 new_scale = scale_value * rsqrt(var_value + (DATA_TYPE4)(*epsilon));
  DATA_TYPE4 new_offset = offset_value - mean_value * new_scale;

  const int pos = ch_blk * width + w_blk;

  DATA_TYPE4 in = READ_IMAGET(input, sampler, (int2)(pos, hb_blk));
  DATA_TYPE4 out = in * new_scale + new_offset;
  WRITE_IMAGET(output, (int2)(pos, hb_blk), out);
}
