#include <common.h>

__kernel void conv_2d(__read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
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
                      __private const int filter_height,
                      __private const int filter_width,
                      __private const int padding_top,
                      __private const int padding_left) {
  const int out_ch_blk = get_global_id(0);
  const int out_w_blk = get_global_id(1);
  const int out_w_blks = get_global_size(1);
  const int out_hb = get_global_id(2);
  const int rounded_in_ch = in_ch_blks * 4;

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#ifdef BIAS
  DATA_TYPE4 out0 =
     READ_IMAGET(bias, sampler, (int2)(out_ch_blk, 0));
  DATA_TYPE4 out1 = out0;
  DATA_TYPE4 out2 = out0;
  DATA_TYPE4 out3 = out0;
#else
  DATA_TYPE4 out0 = 0;
  DATA_TYPE4 out1 = 0;
  DATA_TYPE4 out2 = 0;
  DATA_TYPE4 out3 = 0;
#endif

#if STRIDE == 1
  int in_width0 = out_w_blk - padding_left;
  int in_width1 = in_width0 + out_w_blks;
  int in_width2 = in_width1 + out_w_blks;
  int in_width3 = in_width2 + out_w_blks;
  const int height_idx = (out_hb % out_height) - padding_top;
#else
  int in_width0 = out_w_blk * 2 - padding_left;
  int in_width1 = (out_w_blk + out_w_blks) * 2 - padding_left;
  int in_width2 = (out_w_blk + 2 * out_w_blks) * 2 - padding_left;
  int in_width3 = (out_w_blk + 3 * out_w_blks) * 2 - padding_left;
  const int height_idx = (out_hb % out_height) * 2 - padding_top;
#endif

  const int batch_idx = (out_hb / out_height) * in_height;

  DATA_TYPE4 in0, in1, in2, in3;
  DATA_TYPE4 weights0, weights1, weights2, weights3;
  int in_idx, in_width_idx;
  // Unrolling this loop hurt perfmance
  for (short in_ch_blk = 0; in_ch_blk < in_ch_blks; ++in_ch_blk) {
    for (short hb_idx = 0; hb_idx < filter_height; ++hb_idx) {
      for (short width_idx = 0; width_idx < filter_width; ++width_idx) {

        in_idx = in_ch_blk * in_width;

        int in_hb_value = height_idx + hb_idx;
        in_hb_value = select(in_hb_value + batch_idx,
                             -1,
                             (in_hb_value < 0 || in_hb_value >= in_height));

        int in_width_value;
#define READ_INPUT(i)                                                                \
        in_width_value = in_width##i + width_idx;                                    \
        in_width_value = select(in_idx + in_width_value,                             \
                                -1,                                                  \
                                (in_width_value < 0 || in_width_value >= in_width)); \
        in##i = READ_IMAGET(input, sampler, (int2)(in_width_value, in_hb_value));

        READ_INPUT(0);
        READ_INPUT(1);
        READ_INPUT(2);
        READ_INPUT(3);

#undef READ_INPUT

        int filter_idx = (in_ch_blk << 2) + (hb_idx * filter_width + width_idx) * rounded_in_ch;
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

      }
    }
  }

#ifdef FUSED_RELU
  // TODO relux
  out0 = fmax(out0, 0);
  out1 = fmax(out1, 0);
  out2 = fmax(out2, 0);
  out3 = fmax(out3, 0);
#endif

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

}
