#include <common.h>

// assume channes_per_group mod 4 = 0 && groups mod 4 == 0
__kernel void channel_shuffle(KERNEL_ERROR_PARAMS
                              GLOBAL_WORK_GROUP_SIZE_DIM3
                              __read_only image2d_t input,
                              __private const int groups,
                              __private const int channels_per_group,
                              __write_only image2d_t output) {
  const int group_chan_blk_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (group_chan_blk_idx >= global_size_dim0 || width_idx >= global_size_dim1
      || hb_idx >= global_size_dim2) {
    return;
  }
#endif
  const int width = global_size_dim1;

  const int group_blks = groups / 4;
  const int groups_blks_width = group_blks * width;
  const int channels_per_group_blks = channels_per_group / 4;
  const int channels_per_group_blks_width = channels_per_group_blks * width;

  DATA_TYPE4 in_chan_data0, in_chan_data1, in_chan_data2, in_chan_data3;
  DATA_TYPE4 out_chan_data0, out_chan_data1, out_chan_data2, out_chan_data3;

  int in_x = mad24(group_chan_blk_idx, width, width_idx);
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
