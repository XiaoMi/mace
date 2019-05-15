#include <common.h>

__kernel void space_to_depth(OUT_OF_RANGE_PARAMS
                             GLOBAL_WORK_GROUP_SIZE_DIM3
                             __read_only image2d_t input,
                             __private const int input_height,
                             __private const int input_width,
                             __private const int input_depth,
                             __private const int block_size,
                             __private const int output_height,
                             __private const int output_width,
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

  int out_depth_idx = out_depth_blk_idx << 2;
  int in_depth_idx = out_depth_idx % input_depth;
  int hw_block_size = out_depth_idx / input_depth;
  int bottom_width_idx = mul24(out_width_idx, block_size);
  int in_width_idx = bottom_width_idx + (hw_block_size % block_size);
  int in_height_idx = mad24(out_height_idx, block_size,
      hw_block_size / block_size);

  DATA_TYPE4 in_data = 0;
  int in_x = mad24((in_depth_idx >> 2), input_width, in_width_idx);
  int in_y = mad24(batch_idx, input_height, in_height_idx);
#if defined(DEPTH1)
  int top_width = mul24(out_width_idx + 1, block_size);
  int top_height = mul24(out_height_idx + 1, block_size);
  DATA_TYPE4 t_in_data = 0;
  int t_in_x = in_x;
  int t_width_idx = in_width_idx;
  int t_height_idx = in_height_idx;
  DATA_TYPE *in_data_ptr = (DATA_TYPE*)(&in_data);
  for (int i = 0; i < 4; ++i) {
    t_in_data = READ_IMAGET(input, SAMPLER, (int2)(t_in_x, in_y));
    in_data_ptr[i] = t_in_data.x;
    if (t_width_idx + 1 >= top_width) {
      if (t_height_idx + 1 >= top_height) {
        break;
      }
      t_width_idx = bottom_width_idx;
      t_in_x = mad24((in_depth_idx >> 2), input_width, t_width_idx);
      t_height_idx += 1;
      in_y += 1;
    } else {
      t_width_idx += 1;
      t_in_x += 1;
    }
  }
#elif defined(DEPTH2)
  int top_width = mul24(out_width_idx + 1, block_size);
  int top_height = mul24(out_height_idx + 1, block_size);
  DATA_TYPE4 t_in_data = READ_IMAGET(input, SAMPLER, (int2)(in_x, in_y));
  in_data.x = t_in_data.x;
  in_data.y = t_in_data.y;
  t_in_data = 0;
  if (in_width_idx + 1 >= top_width) {
    if (in_height_idx + 1 < top_height) {
      int t_in_x = mad24((in_depth_idx >> 2), input_width, bottom_width_idx);
      t_in_data = READ_IMAGET(input, SAMPLER, (int2)(t_in_x, in_y + 1));
    }
  } else {
    t_in_data = READ_IMAGET(input, SAMPLER, (int2)(in_x + 1, in_y));
  }
  in_data.z = t_in_data.x;
  in_data.w = t_in_data.y;
#elif defined(DEPTH3)
  int top_width = mul24(out_width_idx + 1, block_size);
  int top_height = mul24(out_height_idx + 1, block_size);
  DATA_TYPE4 in_data0 = READ_IMAGET(input, SAMPLER, (int2)(in_x, in_y));
  DATA_TYPE4 in_data1 = 0;
  if (in_width_idx + 1 >= top_width) {
    if (in_height_idx + 1 < top_height) {
      int t_in_x = mad24((in_depth_idx >> 2), input_width, bottom_width_idx);
      in_data1 = READ_IMAGET(input, SAMPLER, (int2)(t_in_x, in_y + 1));
    }
  } else {
    in_data1 = READ_IMAGET(input, SAMPLER, (int2)(in_x + 1, in_y));
  }
  int left_part_size = 3 - in_depth_idx;
  int right_part_size = 4 - left_part_size;
  switch(left_part_size) {
    case 3:
      in_data.z = in_data0.z;
      in_data.y = in_data0.y;
      in_data.x = in_data0.x;
      break;
    case 2:
      in_data.y = in_data0.z;
      in_data.x = in_data0.y;
      break;
    case 1:
      in_data.x = in_data0.z;
      break;
    default:
      in_data = 0;
  }
  switch(right_part_size) {
    case 3:
      in_data.y = in_data1.x;
      in_data.z = in_data1.y;
      in_data.w = in_data1.z;
      break;
    case 2:
      in_data.z = in_data1.x;
      in_data.w = in_data1.y;
      break;
    case 1:
      in_data.w = in_data1.x;
      break;
    default:
      in_data = 0;
  }
#else
  in_data = READ_IMAGET(input, SAMPLER, (int2)(in_x, in_y));
#endif

  const int out_x = mad24(out_depth_blk_idx, output_width, out_width_idx);
  WRITE_IMAGET(output, (int2)(out_x, out_hb_idx), in_data);
}
