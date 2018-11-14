#include <common.h>

#define MIN_VALUE -FLT_MAX

inline int calculate_avg_block_size(const int pool_size_h,
                                    const int pool_size_w,
                                    const int pos_h,
                                    const int pos_w,
                                    const int h_size,
                                    const int w_size) {
  const int h_start = max(0, pos_h);
  const int w_start = max(0, pos_w);
  const int h_end = min(pos_h + pool_size_h, h_size);
  const int w_end = min(pos_w + pool_size_w, w_size);
  return mul24((h_end - h_start), (w_end - w_start));
}

// Supported data type: half/float
__kernel void pooling(BUFFER_OUT_OF_RANGE_PARAMS
                      GLOBAL_WORK_GROUP_SIZE_DIM3
                      __global IN_DATA_TYPE *input,
                      __private const int in_height,
                      __private const int in_width,
                      __private const int in_chan,
                      __private const int out_height,
                      __private const int out_chan,
                      __private const int pad_top,
                      __private const int pad_left,
                      __private const int stride_h,
                      __private const int stride_w,
                      __private const int kernel_h,
                      __private const int kernel_w,
                      __global OUT_DATA_TYPE *output) {

  const int out_chan_blk_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1);
  const int out_hb_idx = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (out_chan_blk_idx >= global_size_dim0 ||
      out_width_idx >= global_size_dim1 ||
      out_hb_idx >= global_size_dim2) {
    return;
  }
#endif
  const int out_width = global_size_dim1;
  const int in_wc_size = mul24(in_width, in_chan);

  const int batch_idx = out_hb_idx / out_height;
  const int out_height_idx = out_hb_idx - mul24(batch_idx, out_height);
  const int chan_idx = out_chan_blk_idx << 2;
  const int in_height_start = mul24(out_height_idx, stride_h) - pad_top;
  const int in_width_start = mul24(out_width_idx, stride_w) - pad_left;
  int in_offset_base = mad24(mad24(mad24(batch_idx, in_height, in_height_start),
      in_width, in_width_start), in_chan, chan_idx);

#ifdef POOL_AVG
  DATA_TYPE4 res = 0;
  for (int height = 0; height < kernel_h; ++height) {
    int in_height_idx = in_height_start + height;
    if (0 <= in_height_idx && in_height_idx < in_height) {
      int in_offset = mad24(height, in_wc_size, in_offset_base);
      for (int width = 0; width < kernel_w; ++width) {
        int in_width_idx = in_width_start + width;
        if (0 <= in_width_idx && in_width_idx < in_width) {
          DATA_TYPE4 in = CONVERT4(vload4(0, input + in_offset));
          res = res + in;
        }
        in_offset += in_chan;
      }
    }
  }
  const int block_size = calculate_avg_block_size(kernel_h,
                                                  kernel_w,
                                                  in_height_start,
                                                  in_width_start,
                                                  in_height,
                                                  in_width);
  res /= block_size;
#else
  DATA_TYPE4 res = (DATA_TYPE4)(MIN_VALUE);
  for (int height = 0; height < kernel_h; ++height) {
    int in_height_idx = in_height_start + height;
    if (0 <= in_height_idx && in_height_idx < in_height) {
      int in_offset = mad24(height, in_wc_size, in_offset_base);
      for (int width = 0; width < kernel_w; ++width) {
        int in_width_idx = in_width_start + width;
        if (0 <= in_width_idx && in_width_idx < in_width) {
          DATA_TYPE4 in = CONVERT4(vload4(0, input + in_offset));
          res = fmax(res, in);
        }
        in_offset += in_chan;
      }
    }
  }
#endif

  const int out_offset = mad24(mad24(mad24(batch_idx, out_height, out_height_idx),
      out_width, out_width_idx), out_chan, chan_idx);
  int remain_chan = out_chan - chan_idx;
  if (remain_chan < 4) {
    switch(remain_chan) {
      case 3:
        output[out_offset + 2] = res.z;
      case 2:
        output[out_offset + 1] = res.y;
      case 1:
        output[out_offset] = res.x;
    }
    CHECK_OUT_OF_RANGE_FOR_BUFFER(out_offset + remain_chan - 1);
  } else {
    VSTORE4(CONVERT_TO(res, OUT_DATA_TYPE4), output, out_offset);
  }
}
