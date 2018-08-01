#include <common.h>

__kernel void softmax(KERNEL_ERROR_PARAMS
                      GLOBAL_WORK_GROUP_SIZE_DIM3
                      __read_only image2d_t input,
                      __private const int channels,
                      __private const int remain_channels,
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
  const int chan_blks = global_size_dim0 - 1;
  const int width = global_size_dim1;

  int pos = width_idx;
  DATA_TYPE max_value = -FLT_MAX;
  DATA_TYPE sum = 0;
  DATA_TYPE4 data;
  for (short i = 0; i < chan_blks; ++i) {
    data = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
    max_value = max(max_value, data.x);
    max_value = max(max_value, data.y);
    max_value = max(max_value, data.z);
    max_value = max(max_value, data.w);
    pos += width;
  }
  data = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
  switch(remain_channels) {
    case 0:
      max_value = max(max_value, data.w);
    case 1:
      max_value = max(max_value, data.z);
    case 2:
      max_value = max(max_value, data.y);
    case 3:
      max_value = max(max_value, data.x);
  }

  pos = width_idx;
  for (short i = 0; i < chan_blks; ++i) {
    data = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
    data = native_exp(data - max_value);
    sum += data.x;
    sum += data.y;
    sum += data.z;
    sum += data.w;
    pos += width;
  }
  data = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
  data -= max_value;
  switch(remain_channels) {
    case 0:
      sum += native_exp(data.w);
    case 1:
      sum += native_exp(data.z);
    case 2:
      sum += native_exp(data.y);
    case 3:
      sum += native_exp(data.x);
  }

  pos = mad24(chan_blk_idx, width, width_idx);
  data = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
  data -= max_value;
  const int exceeded = mul24(chan_blk_idx, 4) - channels;
  switch(exceeded) {
    case 1:
      data.z = native_exp(data.z) / sum;
    case 2:
      data.y = native_exp(data.y) / sum;
    case 3:
      data.x = native_exp(data.x) / sum;
      break;
    default:
      data = native_exp(data) / sum;
  }

  WRITE_IMAGET(output, (int2)(pos, hb_idx), data);
}
