#include <common.h>

// Supported data type: all
__kernel void space_to_batch(__global DATA_TYPE *space_data_ptr,
                             __global const int *block_shape_ptr,
                             __global const int *paddings_ptr,
                             __private const int space_batch,
                             __private const int space_channel,
                             __private const int space_height,
                             __private const int space_width,
                             __private const int batch_height,
                             __private const int batch_width,
                             __private const int b2s,
                             __global DATA_TYPE* batch_data_ptr) {
  int batch_idx = get_global_id(0);
  int batch_channel_idx = get_global_id(1);
  int batch_pixel_idx = get_global_id(2);

  const int block_height = block_shape_ptr[0];
  const int block_width = block_shape_ptr[1];
  const int padding_height_start = paddings_ptr[0];
  const int padding_width_start = paddings_ptr[2];

  const int batch_pixel_height_idx = batch_pixel_idx / batch_width;
  const int batch_pixel_width_idx = batch_pixel_idx % batch_width;

  const int block_size = block_height * block_width;
  const int space_idx = batch_idx / block_size;
  const int remaining_batch_idx = batch_idx % block_size;
  int space_pixel_height_idx = (remaining_batch_idx / block_width) +
                               batch_pixel_height_idx * block_height;
  int space_pixel_width_idx = (remaining_batch_idx % block_width) +
                              batch_pixel_width_idx * block_width;

  const int batch_data_offset = batch_idx * (space_channel * batch_height * batch_width) +
                                (batch_channel_idx * batch_height * batch_width) +
                                batch_pixel_height_idx * batch_width +
                                batch_pixel_width_idx;

  space_pixel_height_idx -= padding_height_start;
  space_pixel_width_idx -= padding_width_start;
  const int space_data_offset = space_idx * (space_channel * space_height * space_width) +
                                (batch_channel_idx * space_height * space_width) +
                                space_pixel_height_idx * space_width +
                                space_pixel_width_idx;
  if (space_pixel_height_idx < 0 || space_pixel_height_idx >= space_height ||
      space_pixel_width_idx < 0 || space_pixel_width_idx >= space_width) {
    if (!b2s) {
        *(batch_data_ptr + batch_data_offset) = 0;
    }
  } else {
    if (b2s) {
      *(space_data_ptr + space_data_offset) = *(batch_data_ptr + batch_data_offset);
    } else {
      *(batch_data_ptr + batch_data_offset) = *(space_data_ptr + space_data_offset);
    }
  }
}
