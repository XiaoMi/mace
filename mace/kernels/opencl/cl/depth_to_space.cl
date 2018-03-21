#include <common.h>

__kernel void depth_to-space(__read_only image2d_t input,
                      __private const int block_size,
                      __private const int batch_size,
                      __private const int input_height,
                      __private const int input_width,
                      __private const int input_depth,
                      __private const int output_height,
                      __private const int output_width,
                      __private const int output_depth,
                      __write_only image2d_t output) {
  const int ch_blk = get_global_id(0);
  const int w = get_global_id(1);
  const int hb = get_global_id(2);
  const int width = get_global_size(1);

  const int out_idx = mad24(ch_blk, width, w);
  
  const int d = out_idx % output_depth;
  const int out_idx2 = out_idx / output_depth;
  const int w = out_idx2 % output_width
  
  for (short g_blk = 0; g_blk < group_blks; ++g_blk) {
    // fetch 4 groups, for each group fetch 4 channels
    in_chan_data0 = READ_IMAGET(input, SAMPLER, (int2)(in_x, hb_idx));
    in_x += channels_per_group_blks_width;

    in_chan_data1 = READ_IMAGET(input, SAMPLER, (int2)(in_x, hb_idx));
    in_x += channels_per_group_blks_width;

    in_chan_data2 = READ_IMAGET(input, SAMPLER, (int2)(in_x, hb_idx));
    in_x += channels_per_group_blks_width;

    in_chan_data3 = READ_IMAGET(input, SAMPLER, (int2)(in_x, hb_idx));
    in_x += channels_per_group_blks_width;

    out_chan_data0 = (DATA_TYPE4)(in_chan_data0.x, in_chan_data1.x, in_chan_data2.x, in_chan_data3.x);
    out_chan_data1 = (DATA_TYPE4)(in_chan_data0.y, in_chan_data1.y, in_chan_data2.y, in_chan_data3.y);
    out_chan_data2 = (DATA_TYPE4)(in_chan_data0.z, in_chan_data1.z, in_chan_data2.z, in_chan_data3.z);
    out_chan_data3 = (DATA_TYPE4)(in_chan_data0.w, in_chan_data1.w, in_chan_data2.w, in_chan_data3.w);

    int out_x = mad24(mad24(group_chan_blk_idx, groups, g_blk), width, width_idx);
    WRITE_IMAGET(output, (int2)(out_x, hb_idx), out_chan_data0);
    out_x += groups_blks_width;

    WRITE_IMAGET(output, (int2)(out_x, hb_idx), out_chan_data1);
    out_x += groups_blks_width;

    WRITE_IMAGET(output, (int2)(out_x, hb_idx), out_chan_data2);
    out_x += groups_blks_width;

    WRITE_IMAGET(output, (int2)(out_x, hb_idx), out_chan_data3);
  }
}
