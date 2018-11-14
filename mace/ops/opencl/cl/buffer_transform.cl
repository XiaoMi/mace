#include <common.h>

__kernel void pad_input(BUFFER_OUT_OF_RANGE_PARAMS
                        GLOBAL_WORK_GROUP_SIZE_DIM2
                        __global IN_DATA_TYPE *input,
                        __private const int in_height,
                        __private const int in_width,
                        __private const int in_chan,
                        __private const int padded_height,
                        __private const int padded_width,
                        __private const int padded_chan,
                        __private const int pad_top,
                        __private const int pad_left,
                        __global DATA_TYPE *output) {
  const int padded_wc_blk_idx = get_global_id(0);
  const int padded_hb_idx = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (padded_wc_blk_idx >= global_size_dim0 ||
      padded_hb_idx >= global_size_dim1) {
    return;
  }
#endif
  const int padded_chan_blk = (padded_chan + 3) >> 2;
  const int padded_width_idx = padded_wc_blk_idx / padded_chan_blk;
  const int padded_chan_blk_idx =
      padded_wc_blk_idx - mul24(padded_width_idx, padded_chan_blk);
  const int batch_idx = padded_hb_idx / padded_height;
  const int padded_height_idx =
      padded_hb_idx - mul24(batch_idx, padded_height);
  const int padded_chan_idx = padded_chan_blk_idx << 2;
  const int in_height_idx = padded_height_idx - pad_top;
  const int in_width_idx = padded_width_idx - pad_left;
  const int padded_offset = mad24(mad24(mad24(batch_idx, padded_height, padded_height_idx),
      padded_width, padded_width_idx), padded_chan, padded_chan_idx);
  const int in_offset = mad24(mad24(mad24(batch_idx, in_height, in_height_idx),
      in_width, in_width_idx), in_chan, padded_chan_idx);

  DATA_TYPE4 value = 0;
  if (0 <= in_height_idx && in_height_idx < in_height &&
      0 <= in_width_idx && in_width_idx < in_width) {
    const int remain_chan = in_chan - padded_chan_idx;
    if (remain_chan < 4) {
      switch (remain_chan) {
        case 3:
          value.z = CONVERT(input[in_offset + 2]);
        case 2:
          value.y = CONVERT(input[in_offset + 1]);
        case 1:
          value.x = CONVERT(input[in_offset]);
      }
    } else {
      value = CONVERT4(vload4(0, input + in_offset));
    }
  }
  vstore4(value, 0, output + padded_offset);
  CHECK_OUT_OF_RANGE_FOR_BUFFER(padded_offset + 3);
}

// OIHW -> [H, W, (O+3) / 4, I, 4]
__kernel void transform_conv_filter(BUFFER_OUT_OF_RANGE_PARAMS
                                    GLOBAL_WORK_GROUP_SIZE_DIM3
                                    __global IN_DATA_TYPE *input,  // OIHW
                                    __private const int input_offset,
                                    __global DATA_TYPE *output,
                                    __private const int out_chan,
                                    __private const int in_chan,
                                    __private const int height,
                                    __private const int width,
                                    __private const int inner_size) {
  const int in_chan_idx = get_global_id(0);
  const int out_chan_blk_idx = get_global_id(1);
  const int hw_idx = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (in_chan_idx >= global_size_dim0 ||
      out_chan_blk_idx >= global_size_dim1 ||
      hw_idx >= global_size_dim2) {
    return;
  }
#endif
  const int t_in_chan = global_size_dim0;
  const int out_chan_blk = global_size_dim1;

  const int h_idx = hw_idx / width;
  const int w_idx = hw_idx - mul24(h_idx, width);
  const int out_chan_idx = out_chan_blk_idx << 2;
  const int in_offset = mad24(mad24(mad24(out_chan_idx, in_chan, in_chan_idx),
      height, h_idx), width, w_idx) + input_offset;
  const int out_offset = (mad24(mad24(mad24(h_idx, width, w_idx),
      out_chan_blk, out_chan_blk_idx), t_in_chan, in_chan_idx) << 2);

  DATA_TYPE4 value = 0;
  if (in_chan_idx < in_chan) {
    if (out_chan_idx + 3 < out_chan) {
      value.x = CONVERT(input[in_offset]);
      value.y = CONVERT(input[in_offset + inner_size]);
      value.z = CONVERT(input[in_offset + 2 * inner_size]);
      value.w = CONVERT(input[in_offset + 3 * inner_size]);
    } else {
      const int diff = out_chan - out_chan_idx;
      switch(diff) {
        case 3:
          value.z = CONVERT(input[in_offset + 2 * inner_size]);
        case 2:
          value.y = CONVERT(input[in_offset + inner_size]);
        case 1:
          value.x = CONVERT(input[in_offset]);
      }
    }
  }
  VSTORE4(value, output, out_offset);
}

// MIHW -> [M, (I+3) / 4, H, W, 4]
__kernel void transform_dw_conv_filter(BUFFER_OUT_OF_RANGE_PARAMS
                                       GLOBAL_WORK_GROUP_SIZE_DIM3
                                       __global IN_DATA_TYPE *input,  // MIHW
                                       __private const int input_offset,
                                       __global DATA_TYPE *output,
                                       __private const int in_chan,
                                       __private const int in_hw) {
  const int width_idx = get_global_id(0);
  const int height_idx = get_global_id(1);
  const int in_chan_blk_idx = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (width_idx >= global_size_dim0 ||
      height_idx >= global_size_dim1 ||
      in_chan_blk_idx >= global_size_dim2) {
    return;
  }
#endif
  const int width = global_size_dim0;
  const int height = global_size_dim1;
  const int in_chan_idx = in_chan_blk_idx << 2;

  const int in_offset = mad24(in_chan_idx, in_hw,
      mad24(height_idx, width, width_idx)) + input_offset;
  const int out_offset = mad24(in_chan_blk_idx, in_hw,
      mad24(height_idx, width, width_idx)) << 2;

  DATA_TYPE4 value = 0;
  if (in_chan_idx + 3 < in_chan) {
    value.x = CONVERT(input[in_offset]);
    value.y = CONVERT(input[in_offset + in_hw]);
    value.z = CONVERT(input[in_offset + (in_hw << 1)]);
    value.w = CONVERT(input[in_offset + in_hw + (in_hw << 1)]);
  } else {
    const int diff = in_chan - in_chan_idx;
    switch(diff) {
      case 3:
        value.z = CONVERT(input[in_offset + (in_hw << 1)]);
      case 2:
        value.y = CONVERT(input[in_offset + in_hw]);
      case 1:
        value.x = CONVERT(input[in_offset]);
    }
  }
  VSTORE4(value, output, out_offset);
}

__kernel void transform_arg(BUFFER_OUT_OF_RANGE_PARAMS
                            __private const int global_size_dim0,
                            __global IN_DATA_TYPE *input,
                            __private const int input_offset,
                            __global DATA_TYPE *output,
                            __private int size) {
  const int blk_idx = get_global_id(0);

#ifndef NON_UNIFORM_WORK_GROUP
  if (blk_idx >= global_size_dim0) {
    return;
  }
#endif
  const int idx = blk_idx << 2;
  const int diff = size - idx;
  const int in_idx = idx + input_offset;
  DATA_TYPE4 value = 0;
  if (diff < 4) {
    switch (diff) {
      case 3:
        value.z = CONVERT(input[in_idx + 2]);
      case 2:
        value.y = CONVERT(input[in_idx + 1]);
      case 1:
        value.x = CONVERT(input[in_idx]);
    }
  } else {
    value = CONVERT4(vload4(0, input + in_idx));
  }

  VSTORE4(value, output, idx);
}

__kernel void transform_data_type(BUFFER_OUT_OF_RANGE_PARAMS
                                  __private const int global_size_dim0,
                                  __global IN_DATA_TYPE *input,
                                  __private const int input_offset,
                                  __global DATA_TYPE *output) {
  const int out_idx = get_global_id(0);

#ifndef NON_UNIFORM_WORK_GROUP
  if (out_idx >= global_size_dim0) {
    return;
  }
#endif

  DATA_TYPE4 input_value = CONVERT4(vload4(out_idx, input + input_offset));
  vstore4(input_value, out_idx, output);
}
