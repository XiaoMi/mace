#include <common.h>

__kernel void batch_to_space(OUT_OF_RANGE_PARAMS
                             GLOBAL_WORK_GROUP_SIZE_DIM3
                             __read_only image2d_t batch_data,
                             __write_only image2d_t space_data,
                             __private const int block_height,
                             __private const int block_width,
                             __private const int padding_height,
                             __private const int padding_width,
                             __private const int batch_size,
                             __private const int space_height,
                             __private const int space_width,
                             __private const int batch_height,
                             __private const int batch_width) {
  const int chan_idx = get_global_id(0);
  const int batch_w_idx = get_global_id(1);
  const int batch_hb_idx = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (chan_idx >= global_size_dim0 || batch_w_idx >= global_size_dim1
      || batch_hb_idx >= global_size_dim2) {
    return;
  }
#endif

  const int batch_b_idx = batch_hb_idx / batch_height;
  const int batch_h_idx = batch_hb_idx - mul24(batch_b_idx, batch_height);

  const int block_size = mul24(block_height, block_width);
  const int remaining_batch_idx = batch_b_idx / batch_size;
  const int space_b_idx = batch_b_idx - mul24(remaining_batch_idx, batch_size);
  const int n_h = remaining_batch_idx / block_width;
  const int mod_h = remaining_batch_idx - mul24(n_h, block_width);
  const int space_h_idx =
      mad24(batch_h_idx, block_height, n_h - padding_height);
  const int space_w_idx =
      mad24(batch_w_idx, block_width, mod_h - padding_width);

  if (0 <= space_w_idx && space_w_idx < space_width &&
      0 <= space_h_idx && space_h_idx < space_height) {
    int2 batch_coord = (int2)(mad24(chan_idx, batch_width, batch_w_idx), batch_hb_idx);
    DATA_TYPE4 value = READ_IMAGET(batch_data, SAMPLER, batch_coord);

    int2 space_coord = (int2)(mad24(chan_idx, space_width, space_w_idx),
                              mad24(space_b_idx, space_height, space_h_idx));

    WRITE_IMAGET(space_data, space_coord, value);
  }
}
