#include <common.h>

__kernel void softmax(OUT_OF_RANGE_PARAMS
                      GLOBAL_WORK_GROUP_SIZE_DIM3
                      __read_only image2d_t input,
                      __private const int channels,
                      __private const int remain_channels,
                      __write_only image2d_t output) {
  const int chan_blk_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (chan_blk_idx >= global_size_dim0 || width_idx >= global_size_dim1
      || hb_idx >= global_size_dim2) {
    return;
  }
#endif
  const int chan_blks = global_size_dim0 - 1;
  const int width = global_size_dim1;

  int pos = width_idx;
  short i = 0;
  DATA_TYPE max_value = -FLT_MAX;
  DATA_TYPE sum = 0;
  DATA_TYPE4 data_arr[4];

  DATA_TYPE4 vec_max_value = (DATA_TYPE4)(-FLT_MAX);
  DATA_TYPE4 max_value_arr[2];
  DATA_TYPE scalar_max_arr[2];

  DATA_TYPE4 vec_sum_value = (DATA_TYPE4)(0);
  DATA_TYPE4 sum_value_arr[2];
  DATA_TYPE scalar_sum_arr[2];

  // Reduce max begins
  for (i = 0; i < chan_blks - 3; i += 4) {
    data_arr[0] = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
    data_arr[1] = READ_IMAGET(input, SAMPLER, (int2)(pos + mul24(1, width), hb_idx));
    data_arr[2] = READ_IMAGET(input, SAMPLER, (int2)(pos + mul24(2, width), hb_idx));
    data_arr[3] = READ_IMAGET(input, SAMPLER, (int2)(pos + mul24(3, width), hb_idx));

    max_value_arr[0] = fmax(data_arr[0], data_arr[1]);
    max_value_arr[1] = fmax(data_arr[2], data_arr[3]);
    max_value_arr[0] = fmax(max_value_arr[0], max_value_arr[1]);
    vec_max_value = fmax(max_value_arr[0], vec_max_value);

    pos += width * 4;
  }
  scalar_max_arr[0] = fmax(vec_max_value.x, vec_max_value.y);
  scalar_max_arr[1] = fmax(vec_max_value.z, vec_max_value.w);
  max_value = fmax(scalar_max_arr[0], scalar_max_arr[1]);

  for (; i < chan_blks; ++i) {
    data_arr[0] = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
    scalar_max_arr[0] = fmax(data_arr[0].x, data_arr[0].y);
    scalar_max_arr[1] = fmax(data_arr[0].z, data_arr[0].w);
    scalar_max_arr[0] = fmax(scalar_max_arr[0], scalar_max_arr[1]);
    max_value = fmax(scalar_max_arr[0], max_value);

    pos += width;
  }


  data_arr[0] = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
  switch(remain_channels) {
    case 0:
      max_value = fmax(max_value, data_arr[0].w);
    case 1:
      max_value = fmax(max_value, data_arr[0].z);
    case 2:
      max_value = fmax(max_value, data_arr[0].y);
    case 3:
      max_value = fmax(max_value, data_arr[0].x);
  }
 // Reduce max ends

 // Reduce sum begins
  pos = width_idx;
  for (i = 0; i < chan_blks - 3; i += 4) {
    data_arr[0] = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
    data_arr[1] = READ_IMAGET(input, SAMPLER, (int2)(pos + mul24(1, width), hb_idx));
    data_arr[2] = READ_IMAGET(input, SAMPLER, (int2)(pos + mul24(2, width), hb_idx));
    data_arr[3] = READ_IMAGET(input, SAMPLER, (int2)(pos + mul24(3, width), hb_idx));

    data_arr[0] = native_exp(data_arr[0] - max_value);
    data_arr[1] = native_exp(data_arr[1] - max_value);
    data_arr[2] = native_exp(data_arr[2] - max_value);
    data_arr[3] = native_exp(data_arr[3] - max_value);
    sum_value_arr[0] = data_arr[0] + data_arr[1];
    sum_value_arr[1] = data_arr[2] + data_arr[3];
    sum_value_arr[0] = sum_value_arr[0] + sum_value_arr[1];
    vec_sum_value += sum_value_arr[0];

    pos += width * 4;
  }

  scalar_sum_arr[0] = vec_sum_value.x + vec_sum_value.y;
  scalar_sum_arr[1] = vec_sum_value.z + vec_sum_value.w;
  sum = scalar_sum_arr[0] + scalar_sum_arr[1];

  for (; i < chan_blks; ++i) {
    data_arr[0] = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
    data_arr[0] = native_exp(data_arr[0] - max_value);
    scalar_sum_arr[0] = data_arr[0].x + data_arr[0].y;
    scalar_sum_arr[1] = data_arr[0].z + data_arr[0].w;
    scalar_sum_arr[0] = scalar_sum_arr[0] + scalar_sum_arr[1];
    sum += scalar_sum_arr[0];

    pos += width;
  }
  data_arr[0] = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
  data_arr[0] -= max_value;
  switch(remain_channels) {
    case 0:
      sum += native_exp(data_arr[0].w);
    case 1:
      sum += native_exp(data_arr[0].z);
    case 2:
      sum += native_exp(data_arr[0].y);
    case 3:
      sum += native_exp(data_arr[0].x);
  }
 // Reduce sum ends

  pos = mad24(chan_blk_idx, width, width_idx);
  data_arr[0] = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
  data_arr[0] -= max_value;
  const int exceeded = mul24(chan_blk_idx, 4) - channels;
  switch(exceeded) {
    case 1:
      data_arr[0].z = native_exp(data_arr[0].z) / sum;
#ifdef USE_LOG
      data_arr[0].z = native_log(data_arr[0].z);
#endif
    case 2:
      data_arr[0].y = native_exp(data_arr[0].y) / sum;
#ifdef USE_LOG
      data_arr[0].y = native_log(data_arr[0].y);
#endif
    case 3:
      data_arr[0].x = native_exp(data_arr[0].x) / sum;
#ifdef USE_LOG
      data_arr[0].x = native_log(data_arr[0].x);
#endif
      break;
    default:
      data_arr[0] = native_exp(data_arr[0]) / sum;
#ifdef USE_LOG
      data_arr[0] = native_log(data_arr[0]);
#endif
  }

  WRITE_IMAGET(output, (int2)(pos, hb_idx), data_arr[0]);
}
