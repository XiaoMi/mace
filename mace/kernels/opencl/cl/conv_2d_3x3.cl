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

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  VEC_DATA_TYPE(DATA_TYPE, 4) out[5] = {0};
#ifdef BIAS
  out[0] =
      CMD_TYPE(read_image, CMD_DATA_TYPE)(bias, sampler, (int2)(out_ch_blk, 0));
  out[1] = out[0];
  out[2] = out[0];
  out[3] = out[0];
  out[4] = out[0];
#endif

  int w[5];
  w[0] = out_w_blk - padding_left;
  w[1] = w[0] + out_w_blks;
  w[2] = w[1] + out_w_blks;
  w[3] = w[2] + out_w_blks;
  w[4] = w[3] + out_w_blks;

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

  VEC_DATA_TYPE(DATA_TYPE, 4) in[5];
  VEC_DATA_TYPE(DATA_TYPE, 4) weights[4];
  int in_idx, hb_idx, width_idx, in_width_idx;
  // Unrolling this loop hurt perfmance
  for (int in_ch_blk = 0; in_ch_blk < in_ch_blks; ++in_ch_blk) {
    for (int i = 0; i < 9; ++i) {

      in_idx = in_ch_blk * in_width;

      hb_idx = i / 3;
      width_idx = i % 3;
      in_width_idx = w[0] + width_idx;
      // Judge the width border for padding input.
      if (in_width_idx < 0 || in_width_idx >= in_width) {
        in[0] = 0;
      } else {
        in[0] = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, sampler, (int2)(in_idx + in_width_idx, in_hb[hb_idx]));
      }
      in_width_idx = w[1] + width_idx;
      if (in_width_idx < 0 || in_width_idx >= in_width) {
        in[1] = 0;
      } else {
        in[1] = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, sampler, (int2)(in_idx + in_width_idx, in_hb[hb_idx]));
      }
      in_width_idx = w[2] + width_idx;
      if (in_width_idx < 0 || in_width_idx >= in_width) {
        in[2] = 0;
      } else {
        in[2] = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, sampler, (int2)(in_idx + in_width_idx, in_hb[hb_idx]));
      }
      in_width_idx = w[3] + width_idx;
      if (in_width_idx < 0 || in_width_idx >= in_width) {
        in[3] = 0;
      } else {
        in[3] = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, sampler, (int2)(in_idx + in_width_idx, in_hb[hb_idx]));
      }
      in_width_idx = w[4] + width_idx;
      if (in_width_idx < 0 || in_width_idx >= in_width) {
        in[4] = 0;
      } else {
        in[4] = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, sampler, (int2)(in_idx + in_width_idx, in_hb[hb_idx]));
      }


      int filter_idx = (in_ch_blk << 2) + i * rounded_in_ch;
      weights[0] = CMD_TYPE(read_image, CMD_DATA_TYPE)(filter, sampler, (int2)(filter_idx + 0, out_ch_blk));
      weights[1] = CMD_TYPE(read_image, CMD_DATA_TYPE)(filter, sampler, (int2)(filter_idx + 1, out_ch_blk));
      weights[2] = CMD_TYPE(read_image, CMD_DATA_TYPE)(filter, sampler, (int2)(filter_idx + 2, out_ch_blk));
      weights[3] = CMD_TYPE(read_image, CMD_DATA_TYPE)(filter, sampler, (int2)(filter_idx + 3, out_ch_blk));

      // Will prefetch L2 improve performance? How to pretch image data?

      // Interleaving load and mul does not improve performance as expected
      out[0] += in[0].x * weights[0];
      out[0] += in[0].y * weights[1];
      out[0] += in[0].z * weights[2];
      out[0] += in[0].w * weights[3];

      out[1] += in[1].x * weights[0];
      out[1] += in[1].y * weights[1];
      out[1] += in[1].z * weights[2];
      out[1] += in[1].w * weights[3];

      out[2] += in[2].x * weights[0];
      out[2] += in[2].y * weights[1];
      out[2] += in[2].z * weights[2];
      out[2] += in[2].w * weights[3];

      out[3] += in[3].x * weights[0];
      out[3] += in[3].y * weights[1];
      out[3] += in[3].z * weights[2];
      out[3] += in[3].w * weights[3];

      out[4] += in[4].x * weights[0];
      out[4] += in[4].y * weights[1];
      out[4] += in[4].z * weights[2];
      out[4] += in[4].w * weights[3];
    }
  }

  const int out_x_base = out_ch_blk * out_width;
  CMD_TYPE(write_image, CMD_DATA_TYPE)(output,
                                       (int2)(out_x_base + w[0] + padding_left, out_hb),
                                       out[0]);

  w[1] += padding_left;
  if (w[1] >= out_width) return;
  CMD_TYPE(write_image, CMD_DATA_TYPE)(output,
                                       (int2)(out_x_base + w[1], out_hb),
                                       out[1]);

  w[2] += padding_left;
  if (w[2] >= out_width) return;
  CMD_TYPE(write_image, CMD_DATA_TYPE)(output,
                                       (int2)(out_x_base + w[2], out_hb),
                                       out[2]);

  w[3] += padding_left;
  if (w[3] >= out_width) return;
  CMD_TYPE(write_image, CMD_DATA_TYPE)(output,
                                       (int2)(out_x_base + w[3], out_hb),
                                       out[3]);

  w[4] += padding_left;
  if (w[4] >= out_width) return;
  CMD_TYPE(write_image, CMD_DATA_TYPE)(output,
                                       (int2)(out_x_base + w[4], out_hb),
                                       out[4]);
}
