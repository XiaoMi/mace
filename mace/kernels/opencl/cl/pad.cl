#include <common.h>

__kernel void pad(KERNEL_ERROR_PARAMS
                  GLOBAL_WORK_GROUP_SIZE_DIM3
                  __read_only image2d_t input,
                  __write_only image2d_t output,
                  __private const float constant_value,
                  __private const int input_height,
                  __private const int input_width,
                  __private const int output_height,
                  __private const int height_padding,
                  __private const int width_padding) {
  const int chan_blk_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);
  const int batch_idx = hb_idx / output_height;
  const int height_idx = hb_idx % output_height;
  const int input_padded_height = input_height + height_padding;
  const int input_padded_width = input_width + width_padding;

#ifndef NON_UNIFORM_WORK_GROUP
  if (chan_blk_idx >= global_size_dim0 || width_idx >= global_size_dim1
      || hb_idx >= global_size_dim2) {
    return;
  }
#endif
  const int width = global_size_dim1;

  DATA_TYPE4 data = constant_value;
  if ((height_padding <= height_idx && height_idx < input_padded_height) &&
      (width_padding <= width_idx && width_idx < input_padded_width)) {
    const int in_hb_idx = mad24(batch_idx, input_height,
                                height_idx - height_padding);
    data = READ_IMAGET(input,
                       SAMPLER,
                       (int2)(mad24(chan_blk_idx, input_width,
                                    width_idx - width_padding),
                              in_hb_idx));
  }

  const int pos = mad24(chan_blk_idx, width, width_idx);

  WRITE_IMAGET(output, (int2)(pos, hb_idx), data);
}
