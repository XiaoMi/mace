#include <common.h>

DATA_TYPE4 compute_total(__read_only image2d_t input, const int hb_base,
                         const int chan_blks, const int width, const int height,
                         const int hb_idx, const int chan_blk_idx) {
  DATA_TYPE4 total = 0.0f;
#if PARAM_AXIS == 1
  const int wc_blks = mul24(width, chan_blks);

  for (int h_idx = hb_base; h_idx < hb_base + height; ++h_idx) {
    for (int pos = 0; pos < wc_blks; ++pos) {
      DATA_TYPE4 in_data = READ_IMAGET(input, SAMPLER, (int2)(pos, h_idx));
#if PARAM_P == 1
      total += fabs(in_data);
#else
      total = mad(in_data, in_data, total);
#endif
    }
  }
  DATA_TYPE total_all = total.x + total.y + total.z + total.w;
  total = (DATA_TYPE4){total_all, total_all, total_all, total_all};
#elif PARAM_AXIS == 2
  for (int h_idx = hb_base; h_idx < hb_base + height; ++h_idx) {
    for (int w_idx = 0; w_idx < width; ++w_idx) {
      int pos = mad24(chan_blk_idx, width, w_idx);
      DATA_TYPE4 in_data = READ_IMAGET(input, SAMPLER, (int2)(pos, h_idx));
#if PARAM_P == 1
        total = total + fabs(in_data);
#else
        total = mad(in_data, in_data, total);
#endif
    }
  }
#elif PARAM_AXIS == 3
  for (int w_idx = 0; w_idx < width; ++x) {
    int pos = mad24(chan_blk_idx, width, w_idx);
    DATA_TYPE4 in_data = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
#if PARAM_P == 1
      total = total + fabs(in_data);
#else
      total = mad(in_data, in_data, total);
#endif
  }
#endif

  return total;
}

__kernel void lpnorm(OUT_OF_RANGE_PARAMS
                     GLOBAL_WORK_GROUP_SIZE_DIM3
                     __read_only image2d_t input,
                     __private const int height,
                     __private const float eps,
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

  const int chan_blks = global_size_dim0;
  const int width = global_size_dim1;
  const int pos = mad24(chan_blk_idx, width, width_idx);
  DATA_TYPE4 in_data = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
  const int hb_base = mul24(hb_idx / height, height);
  DATA_TYPE4 total = compute_total(input, hb_base, chan_blks, width, height,
                                   hb_idx, chan_blk_idx);

#if PARAM_P == 1
    in_data = in_data / (total + eps);
#else
    in_data = in_data / (sqrt(total) + eps);
#endif
  WRITE_IMAGET(output, (int2)(pos, hb_idx), in_data);
}
