#include <common.h>

__kernel void depth_to_space(OUT_OF_RANGE_PARAMS
                             GLOBAL_WORK_GROUP_SIZE_DIM3
                             __read_only image2d_t input,
                             __private const int block_size,
                             __private const int input_hb,
                             __private const int input_width,
                             __private const int input_depth_blocks,
                             __private const int output_width,
                             __private const int output_depth_blocks,
                             __write_only image2d_t output) {
  const int out_d = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_hb = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (out_d >= global_size_dim0 || out_w >= global_size_dim1
      || out_hb >= global_size_dim2) {
    return;
  }
#endif

  const int out_pos = mad24(out_d, output_width, out_w);

  const int in_hb = out_hb / block_size;
  const int offset_h = out_hb - mul24(in_hb, block_size);
  const int in_w = out_w / block_size;
  const int offset_w = out_w - mul24(in_w, block_size);
  const int offset_d = mul24(mad24(offset_h, block_size, offset_w), output_depth_blocks);
  const int in_d = out_d + offset_d;

  if (in_hb >= input_hb || in_w >= input_width || in_d >= input_depth_blocks) {
    return;
  }

  const int in_pos = mad24(in_d, input_width, in_w);
  DATA_TYPE4 in_data = READ_IMAGET(input, SAMPLER, (int2)(in_pos, in_hb));

  WRITE_IMAGET(output, (int2)(out_pos, out_hb), in_data);
}
