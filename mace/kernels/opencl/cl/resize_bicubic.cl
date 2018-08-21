#include <common.h>
//#include <cstdio>

const int kTableSize = (1 << 10);

inline float ComputeCoeffs(int i) {
//    const int kTableSize = (1 << 10);
  const float A = -0.75;
  float x = (i / 2) * 1.0 / kTableSize;
  if (i % 2 == 0){
    float coeff = ((A + 2) * x - (A + 3)) * x * x + 1;
    return coeff;
  }
  else {
    x += 1.0;
    float coeff = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    return coeff;
  }
}

//#define GET_COEFFS(coeffs_tab, i) coeffs_tab[i]
//float getCoeffs(const float *coeffs_tab, int i) {
//    return coeffs_tab[i];
//}

#define BOUND(val, limit) min(limit - 1, max(0, val))
//int Bound(int val, int limit) {
//    return min(limit - 1, max(0, val));
//}

//float4 GetWeights(const float* coeffs_tab, float scale, int out_loc, int limit) {
//    const int in_loc = scale * out_loc;
//    const float delta = scale * out_loc - in_loc;
//    const int offset = delta * kTableSize + 0.5; //lrintf not found in opencl;
//    float4 weights = {getCoeffs(coeffs_tab, offset * 2 + 1),
//                      getCoeffs(coeffs_tab, offset * 2),
//                      getCoeffs(coeffs_tab, (kTableSize - offset) * 2),
//                      getCoeffs(coeffs_tab, (kTableSize - offset) * 2 + 1)};
//    return weights;
//}
//
//int4 GetIndices(float scale, int out_loc, int limit) {
//    const int in_loc = scale * out_loc;
//    const float delta = scale * out_loc - in_loc;
//    const int offset = delta * kTableSize + 0.5; //lrintf not found in opencl
//    int4 indices = {Bound(in_loc - 1, limit), Bound(in_loc, limit),
//                    Bound(in_loc + 1, limit), Bound(in_loc + 2, limit)};
//    return indices;
//}

__kernel void resize_bicubic_nocache(KERNEL_ERROR_PARAMS
                                     GLOBAL_WORK_GROUP_SIZE_DIM3
                                     __read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
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

  //begin resize bicubic
  const int h_in_loc = height_scale * h;
  const float h_delta = height_scale * h - h_in_loc;
  const int h_offset = h_delta * kTableSize + 0.5; //lrintf not found in opencl;

  const int w_in_loc = width_scale * w;
  const float w_delta = width_scale * w - w_in_loc;
  const int w_offset = w_delta * kTableSize + 0.5; //lrintf not found in opencl;

  float4 y_weights = {ComputeCoeffs(h_offset * 2 + 1),
                      ComputeCoeffs(h_offset * 2),
                      ComputeCoeffs((kTableSize - h_offset) * 2),
                      ComputeCoeffs((kTableSize - h_offset) * 2 + 1)};
  int4 y_indices = {BOUND(h_in_loc - 1, in_height), BOUND(h_in_loc, in_height),
                    BOUND(h_in_loc + 1, in_height), BOUND(h_in_loc + 2, in_height)};
  float4 x_weights = {ComputeCoeffs(w_offset * 2 + 1),
                      ComputeCoeffs(w_offset * 2),
                      ComputeCoeffs((kTableSize - w_offset) * 2),
                      ComputeCoeffs((kTableSize - w_offset) * 2 + 1)};
  int4 x_indices = {BOUND(w_in_loc - 1, in_width), BOUND(w_in_loc, in_width),
                    BOUND(w_in_loc + 1, in_width), BOUND(w_in_loc + 2, in_width)};

  float4 coeffs0 = {0, 0, 0, 0};
  float4 coeffs1 = {0, 0, 0, 0};
  float4 coeffs2 = {0, 0, 0, 0};
  float4 coeffs3 = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i) {
    int y_index = y_indices.s0;
    if ( i == 1 ) { y_index = y_indices.s1; }
    if ( i == 2 ) { y_index = y_indices.s2; }
    if ( i == 3 ) { y_index = y_indices.s3; }
    DATA_TYPE4 data0 = READ_IMAGET(input, SAMPLER,
                                   (int2)(in_w_offset + x_indices.s0, in_h_offset + y_index));
    DATA_TYPE4 data1 = READ_IMAGET(input, SAMPLER,
                                   (int2)(in_w_offset + x_indices.s1, in_h_offset + y_index));
    DATA_TYPE4 data2 = READ_IMAGET(input, SAMPLER,
                                   (int2)(in_w_offset + x_indices.s2, in_h_offset + y_index));
    DATA_TYPE4 data3 = READ_IMAGET(input, SAMPLER,
                                   (int2)(in_w_offset + x_indices.s3, in_h_offset + y_index));

    float4 xw0 = { x_weights.s0, x_weights.s0, x_weights.s0, x_weights.s0 };
    float4 xw1 = { x_weights.s1, x_weights.s1, x_weights.s1, x_weights.s1 };
    float4 xw2 = { x_weights.s2, x_weights.s2, x_weights.s2, x_weights.s2 };
    float4 xw3 = { x_weights.s3, x_weights.s3, x_weights.s3, x_weights.s3 };
    float4 res = { 0, 0, 0, 0 };
    res = mad(xw0, data0, res);
    res = mad(xw1, data1, res);
    res = mad(xw2, data2, res);
    res = mad(xw3, data3, res);
    if ( i == 0 ) { coeffs0 = res; }
    if ( i == 1 ) { coeffs1 = res; }
    if ( i == 2 ) { coeffs2 = res; }
    if ( i == 3 ) { coeffs3 = res; }
  }
  float4 yw0 = { y_weights.s0, y_weights.s0, y_weights.s0, y_weights.s0 };
  float4 yw1 = { y_weights.s1, y_weights.s1, y_weights.s1, y_weights.s1 };
  float4 yw2 = { y_weights.s2, y_weights.s2, y_weights.s2, y_weights.s2 };
  float4 yw3 = { y_weights.s3, y_weights.s3, y_weights.s3, y_weights.s3 };
  DATA_TYPE4 outdata = { 0, 0, 0, 0 };
  outdata = mad(yw0, coeffs0, outdata);
  outdata = mad(yw1, coeffs1, outdata);
  outdata = mad(yw2, coeffs2, outdata);
  outdata = mad(yw3, coeffs3, outdata);
  const int out_w_offset = mul24(ch_blk, out_width);
  const int out_h_offset = mul24(b, out_height);

  WRITE_IMAGET(output, (int2)(out_w_offset + w, out_h_offset + h), outdata);
  //end bicubic
}



