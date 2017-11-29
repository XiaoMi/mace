#include <common.h>

__kernel void conv_2d_3x3(__read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
                          __read_only image2d_t filter, /* cout%4 * cin * kw * kh, cout/4 */
#ifdef BIAS
                          __read_only image2d_t bias, /* cout%4 * cout/4 */
#endif
                          __write_only image2d_t output,
                          __private const int in_height,
                          __private const int in_width,
                          __private const int in_ch_blks,
                          __private const int out_height,
                          __private const int out_width,
                          __private const int padding_top,
                          __private const int padding_left) {
  const int out_ch_blk = get_global_id(0);
  const int out_w_blk = get_global_id(1);
  const int out_w_blks = get_global_size(1);
  const int out_hb = get_global_id(2);
  const int rounded_in_ch = in_ch_blks * 4;

  DATA_TYPE4 out0 = 0;
  DATA_TYPE4 out1 = 0;
  DATA_TYPE4 out2 = 0;
  DATA_TYPE4 out3 = 0;
  DATA_TYPE4 out4 = 0;

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#ifdef BIAS
  out0 =
     READ_IMAGET(bias, sampler, (int2)(out_ch_blk, 0));
  out1 = out0;
  out2 = out0;
  out3 = out0;
  out4 = out0;
#endif

#define DEFINE_IN_WIDTH(i) \
  in_width##i[1] = in_width##i[0] + 1;                                                              \
  in_width##i[2] = in_width##i[0] + 2;                                                              \
  in_width##i[0] = (in_width##i[0] < 0 || in_width##i[0] >= in_width) ? (INT_MIN) : in_width##i[0]; \
  in_width##i[1] = (in_width##i[1] < 0 || in_width##i[1] >= in_width) ? (INT_MIN) : in_width##i[1]; \
  in_width##i[2] = (in_width##i[2] < 0 || in_width##i[2] >= in_width) ? (INT_MIN) : in_width##i[2];

  int in_width0[3];
  int in_width1[3];
  int in_width2[3];
  int in_width3[3];
  int in_width4[3];
  in_width0[0] = out_w_blk - padding_left;
  in_width1[0] = in_width0[0] + out_w_blks;
  in_width2[0] = in_width1[0] + out_w_blks;
  in_width3[0] = in_width2[0] + out_w_blks;
  in_width4[0] = in_width3[0] + out_w_blks;
  DEFINE_IN_WIDTH(0);

  DEFINE_IN_WIDTH(1);

  DEFINE_IN_WIDTH(2);

  DEFINE_IN_WIDTH(3);

  DEFINE_IN_WIDTH(4);

#undef DEFINE_IN_WIDTH

  const int batch_idx = out_hb / out_height;
  const int height_idx = out_hb % out_height;
  int in_hb[3];
  in_hb[0] = height_idx - padding_top;
  in_hb[1] = in_hb[0] + 1;
  in_hb[2] = in_hb[1] + 1;
  // Judge the height border for padding input.
  in_hb[0] = (in_hb[0] < 0 || in_hb[0] >= in_height) ? -1 : in_hb[0] + batch_idx * in_height;
  in_hb[1] = (in_hb[1] < 0 || in_hb[1] >= in_height) ? -1 : in_hb[1] + batch_idx * in_height;
  in_hb[2] = (in_hb[2] < 0 || in_hb[2] >= in_height) ? -1 : in_hb[2] + batch_idx * in_height;

  const int input_image_width = in_ch_blks * in_width;

  DATA_TYPE4 in0, in1, in2, in3, in4;
  DATA_TYPE4 weights0, weights1, weights2, weights3;
  int in_idx, hb_idx, width_idx, in_width_idx;
  // Unrolling this loop hurt perfmance
  for (short in_ch_blk = 0; in_ch_blk < in_ch_blks; ++in_ch_blk) {
    for (short hb_idx = 0; hb_idx < 3; ++ hb_idx) {
      for (short width_idx = 0; width_idx < 3; ++width_idx) {

        in_idx = in_ch_blk * in_width;

        // Judge the width border for padding input.
        in0 = READ_IMAGET(input, sampler, (int2)(in_idx + in_width0[width_idx], in_hb[hb_idx]));
        in1 = READ_IMAGET(input, sampler, (int2)(in_idx + in_width1[width_idx], in_hb[hb_idx]));
        in2 = READ_IMAGET(input, sampler, (int2)(in_idx + in_width2[width_idx], in_hb[hb_idx]));
        in3 = READ_IMAGET(input, sampler, (int2)(in_idx + in_width3[width_idx], in_hb[hb_idx]));
        in4 = READ_IMAGET(input, sampler, (int2)(in_idx + in_width4[width_idx], in_hb[hb_idx]));

        int filter_idx = (in_ch_blk << 2) + (hb_idx *  3 + width_idx) * rounded_in_ch;
        weights0 = READ_IMAGET(filter, sampler, (int2)(filter_idx + 0, out_ch_blk));
        weights1 = READ_IMAGET(filter, sampler, (int2)(filter_idx + 1, out_ch_blk));
        weights2 = READ_IMAGET(filter, sampler, (int2)(filter_idx + 2, out_ch_blk));
        weights3 = READ_IMAGET(filter, sampler, (int2)(filter_idx + 3, out_ch_blk));

        // Will prefetch L2 improve performance? How to pretch image data?

        // Interleaving load and mul does not improve performance as expected
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

        out4 += in4.x * weights0;
        out4 += in4.y * weights1;
        out4 += in4.z * weights2;
        out4 += in4.w * weights3;
      }
    }
  }

  const int out_x_base = out_ch_blk * out_width;
  int w = out_w_blk;
  WRITE_IMAGET(output,
               (int2)(out_x_base + w, out_hb),
               out0);

  w += out_w_blks;
  if (w >= out_width) return;
  WRITE_IMAGET(output,
               (int2)(out_x_base + w, out_hb),
               out1);

  w += out_w_blks;
  if (w >= out_width) return;
  WRITE_IMAGET(output,
               (int2)(out_x_base + w, out_hb),
               out2);

  w += out_w_blks;
  if (w >= out_width) return;
  WRITE_IMAGET(output,
               (int2)(out_x_base + w, out_hb),
               out3);

  w += out_w_blks;
  if (w >= out_width) return;
  WRITE_IMAGET(output,
               (int2)(out_x_base + w, out_hb),
               out4);
}
