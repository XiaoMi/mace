#include <common.h>

__kernel void depth_to_space_d1_d2(OUT_OF_RANGE_PARAMS
                                   GLOBAL_WORK_GROUP_SIZE_DIM3
                                   __read_only image2d_t input,
                                   __private const int input_height,
                                   __private const int input_width,
                                   __private const int block_size,
                                   __private const int output_height,
                                   __private const int output_width,
                                   __private const int output_depth,
                                   __write_only image2d_t output) {
  const int in_depth_blk_idx = get_global_id(0);
  const int in_width_idx = get_global_id(1);
  const int in_hb_idx = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (in_depth_blk_idx >= global_size_dim0 || in_width_idx >= global_size_dim1
      || in_hb_idx >= global_size_dim2) {
    return;
  }
#endif
  const int batch_idx = in_hb_idx / input_height;
  const int in_height_idx = in_hb_idx - mul24(batch_idx, input_height);

  int in_depth_idx = in_depth_blk_idx << 2;
  int hw_block_size = in_depth_idx / output_depth;
  int out_depth_idx = in_depth_idx - mul24(hw_block_size, output_depth);
  int bottom_width_idx = mul24(in_width_idx, block_size);
  int out_width_idx = bottom_width_idx + (hw_block_size % block_size);
  int out_height_idx = mad24(in_height_idx, block_size,
      hw_block_size / block_size);

  const int in_x = mad24(in_depth_blk_idx, input_width, in_width_idx);
  DATA_TYPE4 in_data = READ_IMAGET(input, SAMPLER, (int2)(in_x, in_hb_idx));

  int out_x_base = mul24((out_depth_idx >> 2), output_width);
  int out_x = out_x_base + out_width_idx;
  int out_y = mad24(batch_idx, output_height, out_height_idx);
#if defined(DEPTH1)
  int top_width = mul24(in_width_idx + 1, block_size);
  int top_height = mul24(in_height_idx + 1, block_size);
  DATA_TYPE4 t_out_data = 0;
  int t_out_x = out_x;
  int t_width_idx = out_width_idx;
  int t_height_idx = out_height_idx;
  DATA_TYPE *in_data_ptr = (DATA_TYPE*)(&in_data);
  for (int i = 0; i < 4; ++i) {
    t_out_data.x = in_data_ptr[i];
    WRITE_IMAGET(output, (int2)(t_out_x, out_y), t_out_data);
    if (t_width_idx + 1 >= top_width) {
      if (t_height_idx + 1 >= top_height) {
        break;
      }
      t_width_idx = bottom_width_idx;
      t_out_x = out_x_base + t_width_idx;
      t_height_idx += 1;
      out_y += 1;
    } else {
      t_width_idx += 1;
      t_out_x += 1;
    }
  }
#elif defined(DEPTH2)
  int top_width = mul24(in_width_idx + 1, block_size);
  int top_height = mul24(in_height_idx + 1, block_size);
  DATA_TYPE4 t_out_data = 0;
  t_out_data.x = in_data.x;
  t_out_data.y = in_data.y;
  WRITE_IMAGET(output, (int2)(out_x, out_y), t_out_data);
  t_out_data.x = in_data.z;
  t_out_data.y = in_data.w;
  if (out_width_idx + 1 >= top_width) {
    if (out_height_idx + 1 < top_height) {
      int t_out_x = out_x_base + bottom_width_idx;
      WRITE_IMAGET(output, (int2)(t_out_x, out_y + 1), t_out_data);
    }
  } else {
    WRITE_IMAGET(output, (int2)(out_x + 1, out_y), t_out_data);
  }
#endif
}

__kernel void depth_to_space(OUT_OF_RANGE_PARAMS
                             GLOBAL_WORK_GROUP_SIZE_DIM3
                             __read_only image2d_t input,
                             __private const int input_height,
                             __private const int input_width,
                             __private const int block_size,
                             __private const int output_height,
                             __private const int output_width,
                             __private const int output_depth,
                             __write_only image2d_t output) {
  const int out_depth_blk_idx = get_global_id(0);
  const int out_width_idx = get_global_id(1);
  const int out_hb_idx = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (out_depth_blk_idx >= global_size_dim0 || out_width_idx >= global_size_dim1
      || out_hb_idx >= global_size_dim2) {
    return;
  }
#endif
  const int batch_idx = out_hb_idx / output_height;
  const int out_height_idx = out_hb_idx - mul24(batch_idx, output_height);

  int in_height_idx = out_height_idx / block_size;
  int height_idx_in_blk = out_height_idx - mul24(in_height_idx, block_size);
  int in_width_idx = out_width_idx / block_size;
  int width_idx_in_blk = out_width_idx - mul24(in_width_idx, block_size);
  int in_depth_idx = mad24(
      mad24(height_idx_in_blk, block_size, width_idx_in_blk),
      output_depth, out_depth_blk_idx << 2);

  int in_depth_blk_idx = in_depth_idx >> 2;
  int in_x = mad24(in_depth_blk_idx, input_width, in_width_idx);
  int in_y = mad24(batch_idx, input_height, in_height_idx);
  DATA_TYPE4 out_data = READ_IMAGET(input, SAMPLER, (int2)(in_x, in_y));

#ifdef DEPTH3
  DATA_TYPE4 t_out_data = out_data;
  int left_part_size = 4 - (in_depth_idx & 0x3);
  switch(left_part_size) {
    case 1:
      out_data.x = t_out_data.w;
      break;
    case 2:
      out_data.x = t_out_data.z;
      out_data.y = t_out_data.w;
      break;
    case 3:
      out_data.x = t_out_data.y;
      out_data.y = t_out_data.z;
      out_data.z = t_out_data.w;
      break;
    case 4:
      out_data.x = t_out_data.x;
      out_data.y = t_out_data.y;
      out_data.z = t_out_data.z;
      break;
    default:
      out_data = 0;
  }
  int right_part_size = 3 - left_part_size;
  if (right_part_size > 0) {
    int in_depth_blks = mul24(mul24(block_size, block_size), 3) >> 2;
    in_x = select(-1, in_x + input_width, in_depth_blk_idx + 1 < in_depth_blks);
    t_out_data = READ_IMAGET(input, SAMPLER, (int2)(in_x, in_y));
    switch (right_part_size) {
      case 2:
        out_data.y = t_out_data.x;
        out_data.z = t_out_data.y;
        break;
      case 1:
        out_data.z = t_out_data.x;
        break;
      default:
        out_data = 0;
    }
  }
  out_data.w = 0;
#endif

  int out_x = mad24(out_depth_blk_idx, output_width, out_width_idx);
  WRITE_IMAGET(output, (int2)(out_x, out_hb_idx), out_data);
}
