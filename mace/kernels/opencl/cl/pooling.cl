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
__kernel void pooling(KERNEL_ERROR_PARAMS
                      GLOBAL_WORK_GROUP_SIZE_DIM3
                      __read_only image2d_t input,
                      __private const int in_height,
                      __private const int in_width,
                      __private const int out_height,
                      __private const int pad_top,
                      __private const int pad_left,
                      __private const int stride_h,
                      __private const int stride_w,
                      __private const int pooling_size_h,
                      __private const int pooling_size_w,
                      __write_only image2d_t output) {

  const int out_chan_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1);
  const int out_hb_idx = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (out_chan_idx >= global_size_dim0 || out_width_idx >= global_size_dim1
      || out_hb_idx >= global_size_dim2) {
    return;
  }
#endif
  const int out_width = global_size_dim1;

  const int batch_idx = mul24((out_hb_idx / out_height), in_height);
  const int in_height_start = mul24((out_hb_idx % out_height), stride_h) - pad_top;
  const int in_width_start = mul24(out_width_idx, stride_w) - pad_left;
  const int in_channel_offset = mul24(out_chan_idx, in_width);


#ifdef POOL_AVG
  DATA_TYPE4 res = 0;
  for (int height = 0; height < pooling_size_h; ++height) {
    int in_height_idx = in_height_start + height;
    in_height_idx = select(batch_idx + in_height_idx,
                       -1,
                       (in_height_idx < 0 || in_height_idx >= in_height));
    for (int width = 0; width < pooling_size_w; ++width) {
      int in_width_idx = in_width_start + width;
      in_width_idx = select(in_channel_offset + in_width_idx,
                            -1,
                            (in_width_idx < 0 || in_width_idx >= in_width));

      DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(in_width_idx, in_height_idx));
      res = res + in;
    }
  }
  const int block_size = calculate_avg_block_size(pooling_size_h,
                                                  pooling_size_w,
                                                  in_height_start, in_width_start,
                                                  in_height, in_width);
  res /= block_size;
#else
  DATA_TYPE4 res = (DATA_TYPE4)(MIN_VALUE);
  for (int height = 0; height < pooling_size_h; ++height) {
    int in_height_idx = in_height_start + height;
    in_height_idx = select(batch_idx + in_height_idx,
                           -1,
                           (in_height_idx < 0 || in_height_idx >= in_height));
    if (in_height_idx != -1) {
      for (int width = 0; width < pooling_size_w; ++width) {
        int in_width_idx = in_width_start + width;
        in_width_idx = select(in_channel_offset + in_width_idx,
                              -1,
                              (in_width_idx < 0 || in_width_idx >= in_width));

        if (in_width_idx != -1) {
          DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(in_width_idx, in_height_idx));
          res = fmax(res, in);
        }
      }
    }
  }
#endif

  const int pos = mad24(out_chan_idx, out_width, out_width_idx);
  WRITE_IMAGET(output, (int2)(pos, out_hb_idx), res);
}
