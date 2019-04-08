#include <common.h>

__kernel void softmax(BUFFER_OUT_OF_RANGE_PARAMS
                      GLOBAL_WORK_GROUP_SIZE_DIM3
                      __global IN_DATA_TYPE *input,
                      __private const int height,
                      __private const int channels,
                      __private const int remain_channels,
                      __global OUT_DATA_TYPE *output) {
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
  const int batch_idx = hb_idx / height;
  const int height_idx = hb_idx - mul24(batch_idx, height);
  const int chan_idx = chan_blk_idx << 2;

  const int offset_base = mul24(mad24(mad24(batch_idx, height, height_idx),
      width, width_idx), channels);
  int in_offset = offset_base;
  DATA_TYPE max_value = -FLT_MAX;
  DATA_TYPE sum = 0;
  DATA_TYPE4 data;
  for (short i = 0; i < chan_blks; ++i) {
    data = CONVERT4(vload4(0, input + in_offset));
    max_value = max(max_value, data.x);
    max_value = max(max_value, data.y);
    max_value = max(max_value, data.z);
    max_value = max(max_value, data.w);
    in_offset += 4;
  }
  switch(remain_channels) {
    case 0:
      max_value = max(max_value, CONVERT(input[in_offset + 3]));
    case 1:
      max_value = max(max_value, CONVERT(input[in_offset + 2]));
    case 2:
      max_value = max(max_value, CONVERT(input[in_offset + 1]));
    case 3:
      max_value = max(max_value, CONVERT(input[in_offset]));
  }

  in_offset = offset_base;
  for (short i = 0; i < chan_blks; ++i) {
    data = CONVERT4(vload4(0, input + in_offset));
    data = native_exp(data - max_value);
    sum += data.x;
    sum += data.y;
    sum += data.z;
    sum += data.w;
    in_offset += 4;
  }
  switch(remain_channels) {
    case 0:
      sum += native_exp(CONVERT(input[in_offset + 3]) - max_value);
    case 1:
      sum += native_exp(CONVERT(input[in_offset + 2]) - max_value);
    case 2:
      sum += native_exp(CONVERT(input[in_offset + 1]) - max_value);
    case 3:
      sum += native_exp(CONVERT(input[in_offset]) - max_value);
  }

  int remain_chan = channels - chan_idx;
  int offset = offset_base + chan_idx;
  if (remain_chan < 4) {
    switch(remain_chan) {
      case 3:
        output[offset + 2] = native_exp(CONVERT(input[offset + 2]) - max_value) / sum;
#ifdef USE_LOG
        output[offset + 2] = native_log(output[offset + 2]);
#endif
      case 2:
        output[offset + 1] = native_exp(CONVERT(input[offset + 1]) - max_value) / sum;
#ifdef USE_LOG
        output[offset + 1] = native_log(output[offset + 1]);
#endif
      case 1:
        output[offset] = native_exp(CONVERT(input[offset]) - max_value) / sum;
#ifdef USE_LOG
        output[offset] = native_log(output[offset]);
#endif
    }
  } else {
    data = CONVERT4(vload4(0, input + offset));
    data = native_exp(data - max_value) / sum;
#ifdef USE_LOG
    data = native_log(data)
#endif
    VSTORE4(CONVERT_TO(data, OUT_DATA_TYPE4), output, offset);
  }
}
