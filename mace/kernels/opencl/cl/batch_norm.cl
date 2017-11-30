#include <common.h>
// Supported data types: half/float
void kernel batch_norm(__read_only image2d_t input,
                       __read_only image2d_t scale,
                       __read_only image2d_t offset,
                       __read_only image2d_t mean,
                       __read_only image2d_t var,
                       global const DATA_TYPE *epsilon,
                       private const int width,
                       __write_only image2d_t output,
                       __local VEC_DATA_TYPE(DATA_TYPE, 4) *new_scale,
                       __local VEC_DATA_TYPE(DATA_TYPE, 4) *new_offset) {
  const int ch_blk = get_global_id(0);
  const int w_blk = get_global_id(1);
  const int hb_blk = get_global_id(2);

  const int local_channel = get_local_id(0);
  const int local_w_idx = get_local_id(1);
  const int local_hb_idx = get_local_id(2);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  if(local_hb_idx == 0 && local_w_idx == 0) {
    VEC_DATA_TYPE(DATA_TYPE, 4) scale4 = CMD_TYPE(read_image, CMD_DATA_TYPE)(scale, sampler, (int2)(ch_blk, 0));
    VEC_DATA_TYPE(DATA_TYPE, 4) offset4 = CMD_TYPE(read_image, CMD_DATA_TYPE)(offset, sampler, (int2)(ch_blk, 0));
    VEC_DATA_TYPE(DATA_TYPE, 4) mean4 = CMD_TYPE(read_image, CMD_DATA_TYPE)(mean, sampler, (int2)(ch_blk, 0));
    VEC_DATA_TYPE(DATA_TYPE, 4) var4 = CMD_TYPE(read_image, CMD_DATA_TYPE)(var, sampler, (int2)(ch_blk, 0));

    new_scale[local_channel] = scale4 * rsqrt(var4 + (VEC_DATA_TYPE(DATA_TYPE, 4))(*epsilon));
    new_offset[local_channel] = offset4 - mean4 * new_scale[local_channel];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  VEC_DATA_TYPE(DATA_TYPE, 4) in[4];
  const int width_pos = w_blk << 2;
  const int pos = ch_blk * width + width_pos;
  if (width_pos + 4 < width) {
    for (int i = 0; i < 4; ++i) {
      in[i] = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, sampler, (int2)(pos + i, hb_blk));
      VEC_DATA_TYPE(DATA_TYPE, 4) res = in[i] * new_scale[local_channel] + new_offset[local_channel];
      CMD_TYPE(write_image, CMD_DATA_TYPE)(output, (int2)(pos + i, hb_blk), res);
    }
  } else {
    for (int i = 0; i < width - width_pos; ++i) {
      in[i] = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, sampler, (int2)(pos + i, hb_blk));
      VEC_DATA_TYPE(DATA_TYPE, 4) res = in[i] * new_scale[local_channel] + new_offset[local_channel];
      CMD_TYPE(write_image, CMD_DATA_TYPE)(output, (int2)(pos + i, hb_blk), res);
    }
  }
}

