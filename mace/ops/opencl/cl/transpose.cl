#include <common.h>

__kernel void nhwc_to_nchw(BUFFER_OUT_OF_RANGE_PARAMS
                           GLOBAL_WORK_GROUP_SIZE_DIM3
                           __global IN_DATA_TYPE *input,
                           __private const int in_height,
                           __private const int in_channel,
                           __global OUT_DATA_TYPE *output) {
  const int in_nh = get_global_id(2);
  const int in_n = in_nh / in_height;
  const int in_h = mad24(-in_n, in_height, in_nh);
  const int in_w = get_global_id(1);
  const int in_ch_blk = get_global_id(0);
  const int in_ch_base = mul24(4, in_ch_blk);

  const int N = global_size_dim2 / in_height;
  const int H = in_height;
  const int W = global_size_dim1;
  const int C = in_channel;

#ifndef NON_UNIFORM_WORK_GROUP
  if (in_nh >= global_size_dim2 ||
      in_w >= global_size_dim1 ||
      in_ch_blk >= global_size_dim0) {
    return;
  }
#endif
  int in_offset = mad24(
    mad24(
      mad24(in_n, H, in_h), W, in_w), C, in_ch_base);

  int out_offset = mad24(
    mad24(
      mad24(in_n, C, in_ch_base), H, in_h), W, in_w);
  IN_DATA_TYPE4 in_vec = (IN_DATA_TYPE4)(0);
  const int left = C - in_ch_base;
  int HW = mul24(H, W);
  if (left < 4) {
    switch (left) {
      case 3:
        output[out_offset + mul24(2, HW)] = input[in_offset + 2];
      case 2:
        output[out_offset + HW] = input[in_offset + 1];
      case 1:
        output[out_offset] = input[in_offset];
    }
  } else {
    in_vec = vload4(0, input + in_offset);
    output[out_offset] = in_vec.x;
    output[out_offset + HW] = in_vec.y;
    output[out_offset + mul24(2, HW)] = in_vec.z;
    output[out_offset + mul24(3, HW)] = in_vec.w;
  }
}


__kernel void nchw_to_nhwc(BUFFER_OUT_OF_RANGE_PARAMS
                           GLOBAL_WORK_GROUP_SIZE_DIM3
                           __global IN_DATA_TYPE *input,
                           __private const int in_channel,
                           __private const int in_width,
                           __global OUT_DATA_TYPE *output) {
  const int in_nc = get_global_id(2);
  const int in_n = in_nc / in_channel;
  const int in_c = mad24(-in_n, in_channel, in_nc);
  const int in_h = get_global_id(1);
  const int in_w_blk = get_global_id(0);
  const int in_w_base = mul24(4, in_w_blk);

  const int N = global_size_dim2 / in_channel;
  const int C = in_channel;
  const int H = global_size_dim1;
  const int W = in_width;

#ifndef NON_UNIFORM_WORK_GROUP
  if (in_nc >= global_size_dim2 ||
      in_h >= global_size_dim1 ||
      in_w_blk >= global_size_dim0) {
    return;
  }
#endif
  int in_offset = mad24(
    mad24(
      mad24(in_n, C, in_c), H, in_h), W, in_w_base);
  int out_offset = mad24(
    mad24(
      mad24(in_n, H, in_h), W, in_w_base), C, in_c);
  IN_DATA_TYPE4 in_vec = (IN_DATA_TYPE4)(0);
  const int left = W - in_w_base;
  if (left < 4) {
    switch (left) {
      case 3:
        output[out_offset + mul24(2, C)] = input[in_offset + 2];
      case 2:
        output[out_offset + C] = input[in_offset + 1];
      case 1:
        output[out_offset] = input[in_offset];
    }
  } else {
    in_vec = vload4(0, input + in_offset);
    output[out_offset] = in_vec.x;
    output[out_offset + C] = in_vec.y;
    output[out_offset + mul24(2, C)] = in_vec.z;
    output[out_offset + mul24(3, C)] = in_vec.w;
  }
}
