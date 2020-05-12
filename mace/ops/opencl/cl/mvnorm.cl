#include <common.h>

DATA_TYPE4 compute_mean_image(image2d_t input, const int height,
                              const int width, const int chan_blks,
                              const int group_blks,
                              const int batch_idx, const int chan_blk_idx) {
  DATA_TYPE4 total = 0.0f;
  DATA_TYPE4 mean = 0.0f;
  const int hb_base = mul24(batch_idx, height);

#ifdef ACROSS_CHANNELS
  const int wc_blks = mul24(width, chan_blks);
  for (int h_idx = hb_base; h_idx < hb_base + height; ++h_idx) {
    for (int pos = 0; pos < wc_blks; ++pos) {
      DATA_TYPE4 in_data = READ_IMAGET(input, SAMPLER, (int2)(pos, h_idx));
      total += in_data;
    }
  }
  DATA_TYPE total_value = total.x + total.y + total.z + total.w;
  DATA_TYPE mean_value =
      total_value / (DATA_TYPE)(mul24(mul24(height, wc_blks), 4));
  mean = (DATA_TYPE4){mean_value, mean_value, mean_value, mean_value};
#else
#ifdef GROUP_CHANNELS
  const int group_base = chan_blk_idx / group_blks * group_blks;
  const int wg_blks_start = mul24(width, group_base);
  const int wg_blks_end = wg_blks_start + group_blks * width;
  for (int h_idx = hb_base; h_idx < hb_base + height; ++h_idx) {
    for (int pos = wg_blks_start; pos < wg_blks_end; ++pos) {
      DATA_TYPE4 in_data = READ_IMAGET(input, SAMPLER, (int2)(pos, h_idx));
      total += in_data;
    }
  }
  DATA_TYPE total_value = total.x + total.y + total.z + total.w;
  const int total_num = mul24(mul24(height, wg_blks_end - wg_blks_start), 4);
  DATA_TYPE mean_value = total_value / (DATA_TYPE)(total_num);
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
#endif  // GROUP_CHANNELS
#endif  // ACROSS_CHANNELS

  return mean;
}

__kernel void mvnorm_compute_mean_value(OUT_OF_RANGE_PARAMS
                                        GLOBAL_WORK_GROUP_SIZE_DIM2
                                        __read_only image2d_t input,
                                        __private const int height,
                                        __private const int width,
                                        __private const int group_blks,
                                        __write_only image2d_t output) {
  const int chan_blk_idx = get_global_id(0);
  const int batch_idx = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (chan_blk_idx >= global_size_dim0 || batch_idx >= global_size_dim1) {
    return;
  }
#endif

  const int chan_blks = global_size_dim0;
  const int batch = global_size_dim1;

  DATA_TYPE4 mean = compute_mean_image(input, height, width, chan_blks,
                                       group_blks, batch_idx, chan_blk_idx);
  WRITE_IMAGET(output, (int2)(chan_blk_idx, batch_idx), mean);
}

__kernel void mvnorm_mean(OUT_OF_RANGE_PARAMS
                          GLOBAL_WORK_GROUP_SIZE_DIM3
                          __read_only image2d_t input,
                          __read_only image2d_t mean_image,  // E(X)
                          __private const int height,
                          __private const int group_blks,
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

  DATA_TYPE4 mean = READ_IMAGET(
      mean_image, SAMPLER, (int2)(chan_blk_idx, hb_idx / height));
  in_data -= mean;
  WRITE_IMAGET(output, (int2)(pos, hb_idx), in_data);
}

// compute the (X - EX)^2
__kernel void mvnorm_vn_step1(OUT_OF_RANGE_PARAMS
                              GLOBAL_WORK_GROUP_SIZE_DIM3
                              __read_only image2d_t input,
                              __read_only image2d_t mean_image,  // E(X)
                              __write_only image2d_t square_image,  // (X - EX)^2
                              __private const int height,
                              __private const int group_blks) {
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
  DATA_TYPE4 mean =
      READ_IMAGET(mean_image, SAMPLER, (int2)(chan_blk_idx, hb_idx / height));
  in_data = in_data - mean;
  DATA_TYPE4 pow_data = in_data * in_data;
  WRITE_IMAGET(square_image, (int2)(pos, hb_idx), pow_data);
}

// compute (X - EX) / (E((X - EX)^2)^0.5 + eps_)
__kernel void mvnorm_vn_step2(OUT_OF_RANGE_PARAMS
                              GLOBAL_WORK_GROUP_SIZE_DIM3
                              __read_only image2d_t input,
                              __read_only image2d_t mean_image,  // E(X)
                              __read_only image2d_t mean_image_sqr,  // E((X - EX)^2)
                              __private const int height,
                              __private const int group_blks,
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

  DATA_TYPE4 mean = READ_IMAGET(
      mean_image, SAMPLER, (int2)(chan_blk_idx, hb_idx / height));
  const int pos = mad24(chan_blk_idx, width, width_idx);
  DATA_TYPE4 in_data = READ_IMAGET(input, SAMPLER, (int2)(pos, hb_idx));
  in_data = in_data - mean;

  DATA_TYPE4 mean_sqr = READ_IMAGET(
      mean_image_sqr, SAMPLER, (int2)(chan_blk_idx, hb_idx / height));;

#ifdef GROUP_CHANNELS
  DATA_TYPE4 norm_data = in_data / sqrt(mean_sqr + eps);
#else
  DATA_TYPE4 norm_data = in_data / (sqrt(mean_sqr) + eps);
#endif

  WRITE_IMAGET(output, (int2)(pos, hb_idx), norm_data);
}
