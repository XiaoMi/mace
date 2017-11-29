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
                          __private const int in_ch_blks,
                          __private const int width) {
  const int out_ch_blk = get_global_id(0);
  const int out_w_blk = get_global_id(1);
  const int out_w_blks = get_global_size(1);
  const int out_hb = get_global_id(2);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  DATA_TYPE4 out[4] = {0};
#ifdef BIAS
  out[0] =
      READ_IMAGET(bias, sampler, (int2)(out_ch_blk, 0));
  out[1] = out[0];
  out[2] = out[0];
  out[3] = out[0];
#endif

  int w[4];
  w[0] = out_w_blk;
  w[1] = w[0] + out_w_blks;
  w[2] = w[1] + out_w_blks;
  w[3] = w[2] + out_w_blks;

  // Unrolling this loop hurt perfmance
  int in_x_base = 0;
  for (int in_ch_blk = 0; in_ch_blk < in_ch_blks; ++in_ch_blk) {
    DATA_TYPE4 in[4];
    in[0] = READ_IMAGET(input, sampler, (int2)(in_x_base + w[0], out_hb));
    if (w[1] < width) {
      // conditional load hurt perf, this branching helps sometimes
      in[1] = READ_IMAGET(input, sampler, (int2)(in_x_base + w[1], out_hb));
      in[2] = READ_IMAGET(input, sampler, (int2)(in_x_base + w[2], out_hb));
      in[3] = READ_IMAGET(input, sampler, (int2)(in_x_base + w[3], out_hb));
    }

    const int filter_x0 = in_ch_blk << 2;
    DATA_TYPE4 weights[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
      weights[c] = READ_IMAGET(filter, sampler, (int2)(filter_x0 + c, out_ch_blk));
    }
    // Will prefetch L2 improve performance? How to pretch image data?

    // Interleaving load and mul does not improve performance as expected
    #pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
      out[wi] += in[wi].x * weights[0];
      out[wi] += in[wi].y * weights[1];
      out[wi] += in[wi].z * weights[2];
      out[wi] += in[wi].w * weights[3];
    }

    in_x_base += width;
  }

#ifdef FUSED_BATCH_NORM
  // batch norm
  DATA_TYPE4 bn_scale_value =
      READ_IMAGET(bn_scale, sampler, (int2)(out_ch_blk, 0));
  DATA_TYPE4 scale[4];
  scale[0] = (DATA_TYPE4)(bn_scale_value.x);
  scale[1] = (DATA_TYPE4)(bn_scale_value.y);
  scale[2] = (DATA_TYPE4)(bn_scale_value.z);
  scale[3] = (DATA_TYPE4)(bn_scale_value.w);
  DATA_TYPE4 bn_offset_value =
      READ_IMAGET(bn_offset, sampler, (int2)(out_ch_blk, 0));
  DATA_TYPE4 offset[4];
  offset[0] = (DATA_TYPE4)(bn_offset_value.x);
  offset[1] = (DATA_TYPE4)(bn_offset_value.y);
  offset[2] = (DATA_TYPE4)(bn_offset_value.z);
  offset[3] = (DATA_TYPE4)(bn_offset_value.w);

  #pragma unroll
  for (int wi = 0; wi < 4; ++wi) {
    out[wi] = out[wi] * scale[wi] + offset[wi];
  }
#endif

#ifdef FUSED_RELU
  #pragma unroll
  for (int wi = 0; wi < 4; ++wi) {
    // TODO relux
    out[wi] = fmax(out[wi], 0);
  }
#endif

  const int out_x_base = out_ch_blk * width;
  WRITE_IMAGET(output, (int2)(out_x_base + w[3], out_hb), out[0]);

  if (w[1] >= width) return;
  WRITE_IMAGET(output, (int2)(out_x_base + w[1], out_hb), out[1]);

  if (w[2] >= width) return;
  WRITE_IMAGET(output, (int2)(out_x_base + w[3], out_hb), out[2]);

  if (w[3] >= width) return;
  WRITE_IMAGET(output, (int2)(out_x_base + w[3], out_hb), out[3]);
}
