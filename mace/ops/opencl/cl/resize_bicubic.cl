#include <common.h>

#if CT_MODE == 1    // HALF_PIXEL, for TensorFlow >= 1.14
inline float coeff_even(float i) {
  float x = i / TABLE_SIZE;
  return (1.5f * x - 2.5f) * x * x + 1.0f;
}

inline float coeff_odd(float i) {
  float x = i / TABLE_SIZE + 1.0f;
  return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
}
#else               // NONE, PYTORCH_HALF_PIXEL
inline float coeff_even(float i) {
  float x = i / TABLE_SIZE;
  return (1.25f * x - 2.25f) * x * x + 1.0f;
}

inline float coeff_odd(float i) {
  float x = i / TABLE_SIZE + 1.0f;
  return ((-0.75f * x + 3.75f) * x - 6.0f) * x + 3.0f;
}
#endif

inline void get_weights_and_indices(float scale, int out_loc, int out_size,
                                    int limit, float4 *weights, int4 *indices) {
#if CT_MODE == 0    // NONE
  const float in = out_loc * scale;
#elif CT_MODE == 1  // HALF_PIXEL
  const float in = ((float)out_loc + 0.5f) * scale - 0.5f;
#elif CT_MODE == 2  // PYTORCH_HALF_PIXEL
  const float in =
      select(0.0f, ((float)out_loc + 0.5f) * scale - 0.5f, out_size > 1);
#endif

  const int in_loc = floor(in);
  const float delta = in - in_loc;
  const int offset = convert_int_rte(delta * TABLE_SIZE);
  const float offset_l = offset;
  const float offset_r = TABLE_SIZE - offset_l;

  *indices = (int4) (in_loc - 1, in_loc, in_loc + 1, in_loc + 2);
  *indices = min(max(*indices, 0), limit - 1);
#if CT_MODE == 1    // HALF_PIXEL, for TensorFlow >= 1.14
  (*weights).s0 =
      select(0.0f, coeff_odd(offset_l), (*indices).s0 == in_loc - 1);
  (*weights).s1 =
      select(0.0f, coeff_even(offset_l), (*indices).s1 == in_loc);
  (*weights).s2 =
      select(0.0f, coeff_even(offset_r), (*indices).s2 == in_loc + 1);
  (*weights).s3 =
      select(0.0f, coeff_odd(offset_r), (*indices).s3 == in_loc + 2);
  const float weight_sum =
        (*weights).s0 + (*weights).s1 + (*weights).s2 + (*weights).s3;
  if (fabs(weight_sum) >= 1000.0f * FLT_MIN) {
    (*weights) /= weight_sum;
  }
#else               // NONE, PYTORCH_HALF_PIXEL
  *weights = (float4) (coeff_odd(offset_l), coeff_even(offset_l),
                       coeff_even(offset_r), coeff_odd(offset_r));
#endif
}

__kernel void resize_bicubic_nocache(OUT_OF_RANGE_PARAMS
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
  const int h = hb - mul24(b, out_height);
  const int in_w_offset = mul24(ch_blk, in_width);
  const int in_h_offset = mul24(b, in_height);

  float4 y_weights, x_weights;
  int4 y_indices, x_indices;
  get_weights_and_indices(height_scale, h, out_height, in_height,
                          &y_weights, &y_indices);
  get_weights_and_indices(width_scale, w, out_width, in_width,
                          &x_weights, &x_indices);

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



