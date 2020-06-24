#include <common.h>

__kernel void extract_image_patches(OUT_OF_RANGE_PARAMS
                                    GLOBAL_WORK_GROUP_SIZE_DIM3
                                    __read_only image2d_t input,
                                    __private const int in_height,
                                    __private const int in_width,
                                    __private const int out_height,
                                    __private const int pad_top,
                                    __private const int pad_left,
                                    __private const int stride_h,
                                    __private const int stride_w,
                                    __private const int kernel_h,
                                    __private const int kernel_w,
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
  const int kernel_size = kernel_h * kernel_w;
  const int in_channel = global_size_dim0 / kernel_size;

  const int n_b = out_hb_idx / out_height;
  const int mod_b = out_hb_idx - mul24(n_b, out_height);
  const int in_batch_base = mul24(n_b, in_height);
  const int in_height_start = mad24(mod_b, stride_h, -pad_top);
  const int in_width_start = mad24(out_width_idx, stride_w, -pad_left);
  const int in_chan_idx = out_chan_idx % in_channel;
  const int in_channel_base = mul24(in_chan_idx, in_width);

  const int kernel_base = out_chan_idx / in_channel;
  const int kernel_h_idx = kernel_base / kernel_w;
  const int kernel_w_idx = kernel_base % kernel_w;

  int in_height_idx = in_height_start + kernel_h_idx;
  in_height_idx = select(in_batch_base + in_height_idx, -1,
                         (in_height_idx < 0 || in_height_idx >= in_height));
  int in_width_idx = in_width_start + kernel_w_idx;
  in_width_idx = select(in_channel_base + in_width_idx, -1,
                        (in_width_idx < 0 || in_width_idx >= in_width));
  const int pos = mad24(out_chan_idx, out_width, out_width_idx);
  if (in_height_idx != -1 && in_width_idx != -1) {
    DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(in_width_idx, in_height_idx));
    WRITE_IMAGET(output, (int2)(pos, out_hb_idx), in);
  } else {
    WRITE_IMAGET(output, (int2)(pos, out_hb_idx), 0);
  }
}
