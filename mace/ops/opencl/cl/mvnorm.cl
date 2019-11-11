#include <common.h>

DATA_TYPE4 compute_mean_image(image2d_t input, const int width_idx,
                              const int hb_idx, const int chan_blks,
                              const int height, const int width) {
  DATA_TYPE4 total = 0.0f;
  DATA_TYPE4 mean = 0.0f;
  const int hb_base = mul24(hb_idx / height, height);
  const int wc_blks = mul24(width, chan_blks);

#ifdef ACROSS_CHANNELS
  for (int h_idx = hb_base; h_idx < hb_base + height; ++h_idx) {
    for (int pos = 0; pos < wc_blks; ++pos) {
      DATA_TYPE4 in_data = READ_IMAGET(input, SAMPLER, (int2)(pos, h_idx));
      total += in_data;
    }
  }
  DATA_TYPE total_value = total.x + total.y + total.z + total.w;
  DATA_TYPE mean_value = total_value / (DATA_TYPE)(mul24(mul24(height, wc_blks), 4));
  mean = (DATA_TYPE4){mean_value, mean_value, mean_value, mean_value};
#else
  for (int h_idx = hb_base; h_idx < hb_base + height; ++h_idx) {
    for (int w_idx = 0; w_idx < width; ++w_idx) {
      int pos = mad24(w_idx, chan_blks, width_idx);
      DATA_TYPE4 in_data = READ_IMAGET(input, SAMPLER, (int2)(pos, h_idx));
      total += in_data;
    }
  }
  mean = total / mul24(height, width);
#endif

  return mean;
}

__kernel void mvnorm_mean(OUT_OF_RANGE_PARAMS
                          GLOBAL_WORK_GROUP_SIZE_DIM3
                          __read_only image2d_t input,
                          __private const int height,
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

  DATA_TYPE4 mean = compute_mean_image(input, width_idx,
                                       hb_idx, chan_blks, height, width);
  in_data -= mean;
  WRITE_IMAGET(output, (int2)(pos, hb_idx), in_data);
}

__kernel void mvnorm_vn_step1(OUT_OF_RANGE_PARAMS
                              GLOBAL_WORK_GROUP_SIZE_DIM3
                              __read_only image2d_t input,
                              __write_only image2d_t mean_image,  // E(X)
                              __write_only image2d_t square_image,  // (X - EX)^2
                              __private const int height) {
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
  DATA_TYPE4 mean = compute_mean_image(input, width_idx,
                                       hb_idx, chan_blks, height, width);
  in_data = in_data - mean;
  DATA_TYPE4 pow_data  = in_data * in_data;
  if (hb_idx == 0 && width_idx == 0) {
    WRITE_IMAGET(mean_image, (int2)(chan_blk_idx, 0), mean);
  }
  WRITE_IMAGET(square_image, (int2)(pos, hb_idx), pow_data);
}


__kernel void mvnorm_vn_step2(OUT_OF_RANGE_PARAMS
                              GLOBAL_WORK_GROUP_SIZE_DIM3
                              __read_only image2d_t input,
                              __read_only image2d_t mean_image,  // E(X)
                              __read_only image2d_t square_image,  // (X - EX)^2
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

  DATA_TYPE4 mean = READ_IMAGET(mean_image, SAMPLER, (int2)(chan_blk_idx, 0));
  const int pos = mad24(chan_blk_idx, width, width_idx);
  DATA_TYPE4 in_data = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
  in_data = in_data - mean;

  DATA_TYPE4 mean_v = compute_mean_image(square_image, width_idx,
                                         hb_idx, chan_blks, height, width);

  DATA_TYPE4 norm_data = in_data / (sqrt(mean_v) + eps);

  WRITE_IMAGET(output, (int2)(pos, hb_idx), norm_data);
}
