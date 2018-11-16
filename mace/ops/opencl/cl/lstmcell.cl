#include <common.h>

__kernel void lstmcell(OUT_OF_RANGE_PARAMS
                       GLOBAL_WORK_GROUP_SIZE_DIM2
                       __read_only image2d_t input,
                       __read_only image2d_t pre_output,
                       __read_only image2d_t weight,
                       __read_only image2d_t bias,
                       __read_only image2d_t pre_cell,
                       __private const float forget_bias,
                       __private const int width,
                       __private const int hidden_units,
                       __private const int in_w_blk,
                       __write_only image2d_t cell,
                       __write_only image2d_t output) {
  const int w_blk_idx = get_global_id(0);
  const int h_idx = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w_blk_idx >= global_size_dim0 || h_idx >= global_size_dim1) return;
#endif

  DATA_TYPE4 in, pre_h;
  DATA_TYPE4 w0, w1, w2, w3;
  int k_offset;
  // concat matmul
  const int pos_x0 = w_blk_idx;
  const int pos_x1 = pos_x0 + global_size_dim0;
  const int pos_x2 = pos_x1 + global_size_dim0;
  const int pos_x3 = pos_x2 + global_size_dim0;

  // bias
  // fc_res0 -> i
  // fc_res1 -> j
  // fc_res2 -> f
  // fc_res3 -> o
  DATA_TYPE4 fc_res0, fc_res1, fc_res2, fc_res3;
  fc_res0 = READ_IMAGET(bias, SAMPLER, (int2)(pos_x0, 0));
  fc_res1 = READ_IMAGET(bias, SAMPLER, (int2)(pos_x1, 0));
  fc_res2 = READ_IMAGET(bias, SAMPLER, (int2)(pos_x2, 0));
  fc_res3 = READ_IMAGET(bias, SAMPLER, (int2)(pos_x3, 0));


  for (short i = 0; i < in_w_blk; ++i) {
    in = READ_IMAGET(input, SAMPLER, (int2)(i, h_idx));

    int k = i << 2;
    w0 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x0, k));
    w1 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x1, k));
    w2 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x2, k));
    w3 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x3, k));
    fc_res0 = mad(in.x, w0, fc_res0);
    fc_res1 = mad(in.x, w1, fc_res1);
    fc_res2 = mad(in.x, w2, fc_res2);
    fc_res3 = mad(in.x, w3, fc_res3);

    k += 1;
    k_offset = select(-1, k, k < width);
    w0 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x0, k_offset));
    w1 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x1, k_offset));
    w2 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x2, k_offset));
    w3 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x3, k_offset));
    fc_res0 = mad(in.y, w0, fc_res0);
    fc_res1 = mad(in.y, w1, fc_res1);
    fc_res2 = mad(in.y, w2, fc_res2);
    fc_res3 = mad(in.y, w3, fc_res3);

    k += 1;
    k_offset = select(-1, k, k < width);
    w0 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x0, k_offset));
    w1 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x1, k_offset));
    w2 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x2, k_offset));
    w3 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x3, k_offset));
    fc_res0 = mad(in.z, w0, fc_res0);
    fc_res1 = mad(in.z, w1, fc_res1);
    fc_res2 = mad(in.z, w2, fc_res2);
    fc_res3 = mad(in.z, w3, fc_res3);

    k += 1;
    k_offset = select(-1, k, k < width);
    w0 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x0, k_offset));
    w1 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x1, k_offset));
    w2 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x2, k_offset));
    w3 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x3, k_offset));
    fc_res0 = mad(in.w, w0, fc_res0);
    fc_res1 = mad(in.w, w1, fc_res1);
    fc_res2 = mad(in.w, w2, fc_res2);
    fc_res3 = mad(in.w, w3, fc_res3);
  }

  for (short i = 0; i < global_size_dim0; ++i) {
    pre_h = READ_IMAGET(pre_output, SAMPLER, (int2)(i, h_idx));
    int k = (i << 2) + width;

    w0 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x0, k));
    w1 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x1, k));
    w2 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x2, k));
    w3 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x3, k));
    fc_res0 = mad(pre_h.x, w0, fc_res0);
    fc_res1 = mad(pre_h.x, w1, fc_res1);
    fc_res2 = mad(pre_h.x, w2, fc_res2);
    fc_res3 = mad(pre_h.x, w3, fc_res3);

    k += 1;
    w0 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x0, k));
    w1 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x1, k));
    w2 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x2, k));
    w3 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x3, k));
    fc_res0 = mad(pre_h.y, w0, fc_res0);
    fc_res1 = mad(pre_h.y, w1, fc_res1);
    fc_res2 = mad(pre_h.y, w2, fc_res2);
    fc_res3 = mad(pre_h.y, w3, fc_res3);

    k += 1;
    w0 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x0, k));
    w1 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x1, k));
    w2 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x2, k));
    w3 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x3, k));
    fc_res0 = mad(pre_h.z, w0, fc_res0);
    fc_res1 = mad(pre_h.z, w1, fc_res1);
    fc_res2 = mad(pre_h.z, w2, fc_res2);
    fc_res3 = mad(pre_h.z, w3, fc_res3);

    k += 1;
    w0 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x0, k));
    w1 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x1, k));
    w2 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x2, k));
    w3 = READ_IMAGET(weight, SAMPLER, (int2)(pos_x3, k));
    fc_res0 = mad(pre_h.w, w0, fc_res0);
    fc_res1 = mad(pre_h.w, w1, fc_res1);
    fc_res2 = mad(pre_h.w, w2, fc_res2);
    fc_res3 = mad(pre_h.w, w3, fc_res3);
  }

  // gate
  DATA_TYPE4 pre_c, c, h;
  pre_c = READ_IMAGET(pre_cell, SAMPLER, (int2)(w_blk_idx, h_idx));
  c = do_sigmoid(fc_res0) * tanh(fc_res1) + do_sigmoid((fc_res2 + (float4)forget_bias)) * pre_c;
  h = do_sigmoid(fc_res3) * tanh(c);

  WRITE_IMAGET(cell, (int2)(w_blk_idx, h_idx), c);
  WRITE_IMAGET(output, (int2)(w_blk_idx, h_idx), h);
}
