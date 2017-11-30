#include <common.h>

__kernel void conv_2d_1x1(__read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
                          __read_only image2d_t filter, /* cout%4 * cin, cout/4 */
#ifdef BIAS
                          __read_only image2d_t bias, /* cout%4 * cout/4 */
#endif
#ifdef FUSED_BATCH_NORM
                          __read_only image2d_t bn_scale, /* cout%4 * cout/4 */
                          __read_only image2d_t bn_offset, /* cout%4 * cout/4 */
#endif
                          __write_only image2d_t output,
                          __private const int in_height,
                          __private const int in_width,
                          __private const int in_ch_blks,
                          __private const int height,
                          __private const int width,
                          __private const int padding_top,
                          __private const int padding_left) {
  const int out_ch_blk = get_global_id(0);
  const int out_w_blk = get_global_id(1);
  const int out_w_blks = get_global_size(1);
  const int out_hb = get_global_id(2);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#ifdef BIAS
  DATA_TYPE4 out0 = READ_IMAGET(bias, sampler, (int2)(out_ch_blk, 0));
  DATA_TYPE4 out1 = out0;
  DATA_TYPE4 out2 = out0;
  DATA_TYPE4 out3 = out0;
#else
  DATA_TYPE4 out0 = 0;
  DATA_TYPE4 out1 = 0;
  DATA_TYPE4 out2 = 0;
  DATA_TYPE4 out3 = 0;
#endif

  int4 w;
#if STRIDE == 1
  w.x = out_w_blk - padding_left;
  w.y = w.x + out_w_blks;
  w.z = w.y + out_w_blks;
  w.w = w.z + out_w_blks;
  int out_hb_idx = (out_hb % height) - padding_top;
#else
  w.x = out_w_blk * 2 - padding_left;
  w.y = (out_w_blk + out_w_blks) * 2 - padding_left;
  w.z = (out_w_blk + 2 * out_w_blks) * 2 - padding_left;
  w.w = (out_w_blk + 3 * out_w_blks) * 2 - padding_left;
  int out_hb_idx = (out_hb % height) * 2 - padding_top;
#endif

  w.x = select(w.x, INT_MIN, (w.x < 0 || w.x >= in_width));
  w.y = select(w.y, INT_MIN, (w.y < 0 || w.y >= in_width));
  w.z = select(w.z, INT_MIN, (w.z < 0 || w.z >= in_width));
  w.w = select(w.w, INT_MIN, (w.w < 0 || w.w >= in_width));

  out_hb_idx = select(out_hb_idx + (out_hb / height) * in_height,
                      -1,
                      out_hb_idx >= in_height);

  // Unrolling this loop hurt perfmance
  int in_x_base = 0;
  for (int in_ch_blk = 0; in_ch_blk < in_ch_blks; ++in_ch_blk) {

    DATA_TYPE4 in0 = READ_IMAGET(input, sampler, (int2)(in_x_base + w.x, out_hb_idx));
    DATA_TYPE4 in1 = READ_IMAGET(input, sampler, (int2)(in_x_base + w.y, out_hb_idx));
    DATA_TYPE4 in2 = READ_IMAGET(input, sampler, (int2)(in_x_base + w.z, out_hb_idx));
    DATA_TYPE4 in3 = READ_IMAGET(input, sampler, (int2)(in_x_base + w.w, out_hb_idx));

    const int filter_x0 = in_ch_blk << 2;
    DATA_TYPE4 weights0 = READ_IMAGET(filter, sampler, (int2)(filter_x0, out_ch_blk));
    DATA_TYPE4 weights1 = READ_IMAGET(filter, sampler, (int2)(filter_x0 + 1, out_ch_blk));
    DATA_TYPE4 weights2 = READ_IMAGET(filter, sampler, (int2)(filter_x0 + 2, out_ch_blk));
    DATA_TYPE4 weights3 = READ_IMAGET(filter, sampler, (int2)(filter_x0 + 3, out_ch_blk));
    // Will prefetch L2 improve performance? How to pretch image data?

    out0 += in0.x * weights0;
    out0 += in0.y * weights1;
    out0 += in0.z * weights2;
    out0 += in0.w * weights3;

    out1 += in1.x * weights0;
    out1 += in1.y * weights1;
    out1 += in1.z * weights2;
    out1 += in1.w * weights3;

    out2 += in2.x * weights0;
    out2 += in2.y * weights1;
    out2 += in2.z * weights2;
    out2 += in2.w * weights3;

    out3 += in3.x * weights0;
    out3 += in3.y * weights1;
    out3 += in3.z * weights2;
    out3 += in3.w * weights3;

    in_x_base += in_width;
  }

#ifdef FUSED_BATCH_NORM
  // batch norm
  DATA_TYPE4 bn_scale_value =
      READ_IMAGET(bn_scale, sampler, (int2)(out_ch_blk, 0));
  DATA_TYPE4 scale0 = (DATA_TYPE4)(bn_scale_value.x);
  DATA_TYPE4 scale1 = (DATA_TYPE4)(bn_scale_value.y);
  DATA_TYPE4 scale2 = (DATA_TYPE4)(bn_scale_value.z);
  DATA_TYPE4 scale3 = (DATA_TYPE4)(bn_scale_value.w);
  DATA_TYPE4 bn_offset_value =
      READ_IMAGET(bn_offset, sampler, (int2)(out_ch_blk, 0));
  DATA_TYPE4 offset0 = (DATA_TYPE4)(bn_offset_value.x);
  DATA_TYPE4 offset1 = (DATA_TYPE4)(bn_offset_value.y);
  DATA_TYPE4 offset2 = (DATA_TYPE4)(bn_offset_value.z);
  DATA_TYPE4 offset3 = (DATA_TYPE4)(bn_offset_value.w);

  out0 = out0 * scale0 + offset0;
  out1 = out1 * scale1 + offset1;
  out2 = out2 * scale2 + offset2;
  out3 = out3 * scale3 + offset3;
#endif

#ifdef FUSED_RELU
  // TODO relux
  out0 = fmax(out0, 0);
  out1 = fmax(out1, 0);
  out2 = fmax(out2, 0);
  out3 = fmax(out3, 0);
#endif

  const int out_x_base = out_ch_blk * width;
  int out_x_idx = out_w_blk;
  WRITE_IMAGET(output, (int2)(out_x_base + out_x_idx, out_hb), out0);

  out_x_idx += out_w_blks;
  if (out_x_idx >= width) return;
  WRITE_IMAGET(output, (int2)(out_x_base + out_x_idx, out_hb), out1);

  out_x_idx += out_w_blks;
  if (out_x_idx >= width) return;
  WRITE_IMAGET(output, (int2)(out_x_base + out_x_idx, out_hb), out2);

  out_x_idx += out_w_blks;
  if (out_x_idx >= width) return;
  WRITE_IMAGET(output, (int2)(out_x_base + out_x_idx, out_hb), out3);
}
