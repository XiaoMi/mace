#include <common.h>

inline float coeff_even(float i) {
  float x = i / TABLE_SIZE;
  return (1.25f * x - 2.25f) * x * x + 1.0f;
}

inline float coeff_odd(float i) {
  float x = i / TABLE_SIZE + 1.0f;
  return ((-0.75f * x + 3.75f) * x - 6.0f) * x + 3.0f;
}

__kernel void resize_bicubic_nocache(KERNEL_ERROR_PARAMS
                                     GLOBAL_WORK_GROUP_SIZE_DIM3
                                     __read_only image2d_t input,
                                     __write_only image2d_t output,
                                     __private const float height_scale,
                                     __private const float width_scale,
                                     __private const int in_height,
                                     __private const int in_width,
                                     __private const int out_height) {
  const int ch_blk = get_global_id(0);
  const int w = get_global_id(1);
  const int hb = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (ch_blk >= global_size_dim0 || w >= global_size_dim1
      || hb >= global_size_dim2) {
    return;
  }
  const int ch_blks = global_size_dim0;
  const int out_width = global_size_dim1;
#else
  const int ch_blks = get_global_size(0);
  const int out_width = get_global_size(1);
#endif

  const int b = hb / out_height;
  const int h = hb % out_height;

  const float h_in = h * height_scale;
  const float w_in = w * width_scale;

  const int in_w_offset = mul24(ch_blk, in_width);
  const int in_h_offset = mul24(b, in_height);

  const int h_in_loc = (int)h_in;
  const float h_delta = h_in - h_in_loc;
  const int h_offset = h_delta * TABLE_SIZE + 0.5f;

  const int w_in_loc = (int)w_in;
  const float w_delta = w_in - w_in_loc;
  const int w_offset = w_delta * TABLE_SIZE + 0.5f;

  const float h_offset_l = h_offset;
  const float h_offset_r = TABLE_SIZE - h_offset_l;
  float4 y_weights = {coeff_odd(h_offset_l), coeff_even(h_offset_l),
                      coeff_even(h_offset_r), coeff_odd(h_offset_r)};
  int4 y_indices = {h_in_loc - 1, h_in_loc, h_in_loc + 1, h_in_loc + 2};
  y_indices = min(max(y_indices, 0), in_height - 1);

  const float w_offset_l = w_offset;
  const float w_offset_r = TABLE_SIZE - w_offset_l;
  float4 x_weights = {coeff_odd(w_offset_l), coeff_even(w_offset_l),
                      coeff_even(w_offset_r), coeff_odd(w_offset_r)};
  int4 x_indices = {w_in_loc - 1, w_in_loc, w_in_loc + 1, w_in_loc + 2};
  x_indices = min(max(x_indices, 0), in_width - 1);

  float4 coeffs0 = 0, coeffs1 = 0, coeffs2 = 0, coeffs3 = 0;
  for (int i = 0; i < 4; ++i) {
    int y_index = y_indices.s0;
    if ( i == 1 ) { y_index = y_indices.s1; }
    if ( i == 2 ) { y_index = y_indices.s2; }
    if ( i == 3 ) { y_index = y_indices.s3; }
    const int in_h_index = in_h_offset + y_index;
    DATA_TYPE4 data0 = READ_IMAGET(input, SAMPLER,
             (int2)(in_w_offset + x_indices.s0, in_h_index));
    DATA_TYPE4 data1 = READ_IMAGET(input, SAMPLER,
             (int2)(in_w_offset + x_indices.s1, in_h_index));
    DATA_TYPE4 data2 = READ_IMAGET(input, SAMPLER,
             (int2)(in_w_offset + x_indices.s2, in_h_index));
    DATA_TYPE4 data3 = READ_IMAGET(input, SAMPLER,
             (int2)(in_w_offset + x_indices.s3, in_h_index));

    float4 res = 0;
    res = mad(data0, x_weights.s0, res);
    res = mad(data1, x_weights.s1, res);
    res = mad(data2, x_weights.s2, res);
    res = mad(data3, x_weights.s3, res);
    if ( i == 0 ) { coeffs0 = res; }
    if ( i == 1 ) { coeffs1 = res; }
    if ( i == 2 ) { coeffs2 = res; }
    if ( i == 3 ) { coeffs3 = res; }
  }
  DATA_TYPE4 outdata = 0;
  outdata = mad(coeffs0, y_weights.s0, outdata);
  outdata = mad(coeffs1, y_weights.s1, outdata);
  outdata = mad(coeffs2, y_weights.s2, outdata);
  outdata = mad(coeffs3, y_weights.s3, outdata);
  const int out_w_offset = mul24(ch_blk, out_width);
  const int out_h_offset = mul24(b, out_height);

  WRITE_IMAGET(output, (int2)(out_w_offset + w, out_h_offset + h), outdata);
}



