#include <common.h>

DATA_TYPE4 stitch_vector(DATA_TYPE4 left,
                         DATA_TYPE4 right,
                         const int pos,
                         const bool reversed) {
  if (!reversed) {
    switch (pos) {
      case 1:return (DATA_TYPE4)(left.x, right.x, right.y, right.z);
      case 2:return (DATA_TYPE4)(left.x, left.y, right.x, right.y);
      case 3:return (DATA_TYPE4)(left.x, left.y, left.z, right.x);
      default:return (DATA_TYPE4) 0;
    }
  } else {
    switch (pos) {
      case 1:return (DATA_TYPE4)(left.w, right.x, right.y, right.z);
      case 2:return (DATA_TYPE4)(left.z, left.w, right.x, right.y);
      case 3:return (DATA_TYPE4)(left.y, left.z, left.w, right.x);
      default:return (DATA_TYPE4) 0;
    }
  }
}

// Supported data type: half/float
__kernel void concat_channel(KERNEL_ERROR_PARAMS
                             GLOBAL_WORK_GROUP_SIZE_DIM3
                             __read_only image2d_t input0,
                             __read_only image2d_t input1,
                             __private const int input0_chan,
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

  const int input0_chan_blk = (input0_chan + 3) >> 2;

  DATA_TYPE4 data = 0;
#ifdef DIVISIBLE_FOUR
  if (chan_blk_idx + 1 <= input0_chan_blk) {
    data = READ_IMAGET(input0,
                       SAMPLER,
                       (int2)(mad24(chan_blk_idx, width, width_idx), hb_idx));
  } else {
    data = READ_IMAGET(input1,
                       SAMPLER,
                       (int2)(mad24((chan_blk_idx - input0_chan_blk), width, width_idx), hb_idx));
  }
#else
  if (chan_blk_idx + 1 < input0_chan_blk) {
    data = READ_IMAGET(input0,
                       SAMPLER,
                       (int2)(mad24(chan_blk_idx, width, width_idx), hb_idx));
  } else if (chan_blk_idx >= input0_chan_blk) {
    const int in_chan_idx = chan_blk_idx - input0_chan_blk;
    DATA_TYPE4 data0 = READ_IMAGET(input1,
                                   SAMPLER,
                                   (int2)(mad24(in_chan_idx, width, width_idx), hb_idx));
    DATA_TYPE4 data1 = READ_IMAGET(input1,
                                   SAMPLER,
                                   (int2)(mad24((in_chan_idx + 1), width, width_idx), hb_idx));
    data = stitch_vector(data0, data1, input0_chan % 4, true);
  } else {
    DATA_TYPE4 data0 = READ_IMAGET(input0,
                                   SAMPLER,
                                   (int2)(mad24(chan_blk_idx, width, width_idx), hb_idx));
    DATA_TYPE4 data1 = READ_IMAGET(input1,
                                   SAMPLER,
                                   (int2)(width_idx, hb_idx));
    data = stitch_vector(data0, data1, input0_chan % 4, false);
  }
#endif

  const int pos = mad24(chan_blk_idx, width, width_idx);

  WRITE_IMAGET(output, (int2)(pos, hb_idx), data);
}

// Required: All input channels are divisible by 4
__kernel void concat_channel_multi(KERNEL_ERROR_PARAMS
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

  DATA_TYPE4 data = 0;
  data = READ_IMAGET(input,
                     SAMPLER,
                     (int2)(mad24(chan_blk_idx, width, width_idx), hb_idx));

  const int pos = mad24(chan_blk_idx + chan_blk_offset, width, width_idx);

  WRITE_IMAGET(output, (int2)(pos, hb_idx), data);
}

//__kernel void concat_width(__read_only image2d_t input0,
//                           __read_only image2d_t input1,
//                           __private const int input0_width,
//                           __write_only image2d_t output) {
//  const int chan_blk_idx = get_global_id(0);
//  const int width_idx = get_global_id(1);
//  const int width = get_global_size(1);
//  const int hb_idx = get_global_id(2);
//
//  const sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
//
//  DATA_TYPE4 data = 0;
//  if (width_idx < input0_width) {
//    data = READ_IMAGET(input0,
//                       SAMPLER,
//                       (int2)(chan_blk_idx * width + width_idx, hb_idx));
//  } else {
//    data = READ_IMAGET(input1,
//                       SAMPLER,
//                       (int2)(chan_blk_idx * width + (width_idx - input0_width), hb_idx));
//  }
//
//  WRITE_IMAGET(output, (int2)(chan_blk_idx * width + width_idx, hb_idx), data);
//}

