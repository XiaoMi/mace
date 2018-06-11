#include <common.h>

__kernel void reduce_mean(KERNEL_ERROR_PARAMS
                          GLOBAL_WORK_GROUP_SIZE_DIM3
                          __read_only image2d_t input,
                          __local float4* group_sum,
                          __private const int group_size,
                          __private const int partial_len,
                          __private const int remain_index,
                          __private const int batch,
                          __private const int in_height,
                          __private const int in_width,
                          __private const float in_height_r,
                          __private const float in_width_r,
                          __private const int channel_blocks,
                          __write_only image2d_t output) {
  const int i = get_local_id(0);
  const int j = get_local_id(1);
  const int k = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (i >= local_size_dim0 || j >= local_size_dim1 || k >= global_size_dim2)
    return;
  const int dim0_size = local_size_dim0;
#else
  const int dim0_size = get_local_size(0);
#endif
  DATA_TYPE4 tmp = (DATA_TYPE4){0, 0, 0, 0};
  const int index = j * dim0_size + i;
  const int b = k / channel_blocks;
  const int ch = k - b * channel_blocks;

  DATA_TYPE4 in;
  const int valid_part_len = select(partial_len,
                                    partial_len - 1,
                                    remain_index > 0 && index >= remain_index);
  const int full_offset = index * partial_len;
  const int base_offset = select(full_offset,
                                 full_offset - (index - remain_index),
                                 valid_part_len < partial_len);
#pragma unroll
  for (int l = 0; l < valid_part_len; ++l) {
    int offset = base_offset + l;
    int h_id = floor(offset * in_width_r);
    int w_id = offset - h_id * in_width;
    int pos_x = mad24(ch, in_width, w_id);
    int pos_y = mad24(b, in_height, h_id);
    in = READ_IMAGET(input, SAMPLER, (int2)(pos_x, pos_y));
    tmp = tmp + in;
  }
  group_sum[index] = tmp;

#ifdef NON_QUALCOMM_ADRENO
  barrier(CLK_LOCAL_MEM_FENCE);
#endif

  if (i == 0 && j == 0) {
    DATA_TYPE4 out = (DATA_TYPE4){0, 0, 0, 0};
#pragma unroll
    for (int l = 0; l < group_size; ++l) {
      out = out + group_sum[l];
    }
    out = out * in_height_r * in_width_r;
    WRITE_IMAGET(output, (int2)(ch, b), out);
  }
}
