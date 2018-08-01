#include <common.h>

__kernel void conv_2d_1x1(KERNEL_ERROR_PARAMS
                          GLOBAL_WORK_GROUP_SIZE_DIM3
                          __read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
                          __read_only image2d_t filter, /* cout%4 * cin, cout/4 */
#ifdef BIAS
                          __read_only image2d_t bias, /* cout%4 * cout/4 */
#endif
                          __write_only image2d_t output,
                          __private const float relux_max_limit,
                          __private const int in_height,
                          __private const int in_width,
                          __private const int in_ch_blks,
                          __private const int height,
                          __private const int width,
                          __private const int stride) {
  const int out_ch_blk = get_global_id(0);
  const int out_w_blk = get_global_id(1);
  const int out_hb = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (out_ch_blk >= global_size_dim0 || out_w_blk >= global_size_dim1
      || out_hb >= global_size_dim2) {
    return;
  }
#endif
  const int out_w_blks = global_size_dim1;

#ifdef BIAS
  DATA_TYPE4 out0 = READ_IMAGET(bias, SAMPLER, (int2)(out_ch_blk, 0));
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
  int in_width_stride = mul24(out_w_blks, stride);
  w.x = mul24(out_w_blk, stride);
  w.y = w.x + in_width_stride;
  w.z = w.y + in_width_stride;
  w.w = w.z + in_width_stride;
  int out_hb_idx = mul24((out_hb % height), stride);

  w.x = select(w.x, INT_MIN, w.x >= in_width);
  w.y = select(w.y, INT_MIN, w.y >= in_width);
  w.z = select(w.z, INT_MIN, w.z >= in_width);
  w.w = select(w.w, INT_MIN, w.w >= in_width);

  out_hb_idx = select(mad24((out_hb / height), in_height, out_hb_idx),
                      -1,
                      out_hb_idx >= in_height);

  // Unrolling this loop hurt perfmance
  int in_x_base = 0;
  int filter_x_base = 0;
  for (int in_ch_blk = 0; in_ch_blk < in_ch_blks; ++in_ch_blk) {
    DATA_TYPE4 in0 = READ_IMAGET(input, SAMPLER, (int2)(in_x_base + w.x, out_hb_idx));
    DATA_TYPE4 in1 = READ_IMAGET(input, SAMPLER, (int2)(in_x_base + w.y, out_hb_idx));
    DATA_TYPE4 in2 = READ_IMAGET(input, SAMPLER, (int2)(in_x_base + w.z, out_hb_idx));
    DATA_TYPE4 in3 = READ_IMAGET(input, SAMPLER, (int2)(in_x_base + w.w, out_hb_idx));

    DATA_TYPE4 weights0 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_base + 0, out_ch_blk));
    DATA_TYPE4 weights1 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_base + 1, out_ch_blk));
    DATA_TYPE4 weights2 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_base + 2, out_ch_blk));
    DATA_TYPE4 weights3 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_base + 3, out_ch_blk));

    out0 = mad(in0.x, weights0, out0);
    out0 = mad(in0.y, weights1, out0);
    out0 = mad(in0.z, weights2, out0);
    out0 = mad(in0.w, weights3, out0);

    out1 = mad(in1.x, weights0, out1);
    out1 = mad(in1.y, weights1, out1);
    out1 = mad(in1.z, weights2, out1);
    out1 = mad(in1.w, weights3, out1);

    out2 = mad(in2.x, weights0, out2);
    out2 = mad(in2.y, weights1, out2);
    out2 = mad(in2.z, weights2, out2);
    out2 = mad(in2.w, weights3, out2);

    out3 = mad(in3.x, weights0, out3);
    out3 = mad(in3.y, weights1, out3);
    out3 = mad(in3.z, weights2, out3);
    out3 = mad(in3.w, weights3, out3);

    in_x_base += in_width;
    filter_x_base += 4;
  }

#if defined(USE_RELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
  out0 = do_activation(out0, relux_max_limit);
  out1 = do_activation(out1, relux_max_limit);
  out2 = do_activation(out2, relux_max_limit);
  out3 = do_activation(out3, relux_max_limit);
#endif

  const int out_x_base = mul24(out_ch_blk, width);
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
