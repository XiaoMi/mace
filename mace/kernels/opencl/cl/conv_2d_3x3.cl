#include <common.h>

__kernel void conv_2d_3x3(__read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
                          __read_only image2d_t filter, /* cout%4 * cin * kw * kh, cout/4 */
#ifdef BIAS
                          __read_only image2d_t bias, /* cout%4 * cout/4 */
#endif
                          __write_only image2d_t output,
                          __private const int in_height,
                          __private const int in_width,
                          __private const int in_channels,
                          __private const int out_height,
                          __private const int out_width,
                          __private const int padding_top,
                          __private const int padding_left) {
  const int out_ch_blk = get_global_id(0);
  const int out_w_blk = get_global_id(1);
  const int out_w_blks = get_global_size(1);
  const int out_hb = get_global_id(2);
  const int in_ch_blks = (in_channels + 3) / 4;
  const int rounded_in_ch = in_ch_blks * 4;

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  VEC_DATA_TYPE(DATA_TYPE, 4) out[4] = {0};
#ifdef BIAS
  out[0] =
      CMD_TYPE(read_image, CMD_DATA_TYPE)(bias, sampler, (int2)(out_ch_blk, 0));
  out[1] = out[0];
  out[2] = out[0];
  out[3] = out[0];
#endif

  int w[4];
  w[0] = out_w_blk - padding_left;
  w[1] = w[0] + out_w_blks;
  w[2] = w[1] + out_w_blks;
  w[3] = w[2] + out_w_blks;

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

  // Unrolling this loop hurt perfmance
  int idx = 0;
  for (int in_ch_blk = 0; in_ch_blk < in_ch_blks; ++in_ch_blk) {
    VEC_DATA_TYPE(DATA_TYPE, 4) in[36];
    VEC_DATA_TYPE(DATA_TYPE, 4) weights[36];

    int filter_idx = in_ch_blk << 2;
    int in_idx = in_ch_blk * in_width;

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        idx = i * 12 + j * 4;
        int in_width_idx = w[0] + j;
        // Judge the width border for padding input.
        if (in_width_idx < 0 || in_width_idx >= in_width) {
          in[idx + 0] = 0;
        } else {
          in[idx + 0] = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, sampler, (int2)(in_idx + in_width_idx, in_hb[i]));
        }
        in_width_idx = w[1] + j;
        if (in_width_idx < 0 || in_width_idx >= in_width) {
          in[idx + 1] = 0;
        } else {
          in[idx + 1] = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, sampler, (int2)(in_idx + in_width_idx, in_hb[i]));
        }
        in_width_idx = w[2] + j;
        if (in_width_idx < 0 || in_width_idx >= in_width) {
          in[idx + 2] = 0;
        } else {
          in[idx + 2] = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, sampler, (int2)(in_idx + in_width_idx, in_hb[i]));
        }
        in_width_idx = w[3] + j;
        if (in_width_idx < 0 || in_width_idx >= in_width) {
          in[idx + 3] = 0;
        } else {
          in[idx + 3] = CMD_TYPE(read_image, CMD_DATA_TYPE)(input, sampler, (int2)(in_idx + in_width_idx, in_hb[i]));
        }

        weights[idx + 0] = CMD_TYPE(read_image, CMD_DATA_TYPE)(filter, sampler, (int2)(filter_idx + 0, out_ch_blk));
        weights[idx + 1] = CMD_TYPE(read_image, CMD_DATA_TYPE)(filter, sampler, (int2)(filter_idx + 1, out_ch_blk));
        weights[idx + 2] = CMD_TYPE(read_image, CMD_DATA_TYPE)(filter, sampler, (int2)(filter_idx + 2, out_ch_blk));
        weights[idx + 3] = CMD_TYPE(read_image, CMD_DATA_TYPE)(filter, sampler, (int2)(filter_idx + 3, out_ch_blk));

        filter_idx += rounded_in_ch;
      }
    }
    // Will prefetch L2 improve performance? How to pretch image data?

    // Interleaving load and mul does not improve performance as expected
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
      for (int i = 0; i < 9; ++i) {
        out[c] += in[c + i * 4].x * weights[0 + i * 4];
        out[c] += in[c + i * 4].y * weights[1 + i * 4];
        out[c] += in[c + i * 4].z * weights[2 + i * 4];
        out[c] += in[c + i * 4].w * weights[3 + i * 4];
      }
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
}
