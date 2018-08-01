#include <common.h>

// Only multiplier = 1 is supported
__kernel void depthwise_conv2d(KERNEL_ERROR_PARAMS
                               GLOBAL_WORK_GROUP_SIZE_DIM3
                               __read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
                               __read_only image2d_t filter, /* cout%4 * kh * kw * m, cin/4 */
#ifdef BIAS
    __read_only image2d_t bias, /* cout%4 * cout/4 */
#endif
                               __write_only image2d_t output,
                               __private const float relux_max_limit,
                               __private const short in_height,
                               __private const short in_width,
                               __private const short in_ch_blks,
                               __private const short out_height,
                               __private const short out_width,
                               __private const short filter_height,
                               __private const short filter_width,
                               __private const short padding_top,
                               __private const short padding_left,
                               __private const short dilation_h,
                               __private const short dilation_w) {
  const short out_ch_blk = get_global_id(0);
  const short out_w_blk = get_global_id(1);
  const short out_hb = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (out_ch_blk >= global_size_dim0 || out_w_blk >= global_size_dim1
      || out_hb >= global_size_dim2) {
    return;
  }
#endif
  const short out_w_blks = global_size_dim1;

  const short rounded_in_ch = in_ch_blks << 2;
  const short in_ch_blk = out_ch_blk; // multiplier = 1

#ifdef BIAS
  DATA_TYPE4 out0 =
     READ_IMAGET(bias, SAMPLER, (int2)(out_ch_blk, 0));
  DATA_TYPE4 out1 = out0;
  DATA_TYPE4 out2 = out0;
  DATA_TYPE4 out3 = out0;
#else
  DATA_TYPE4 out0 = 0;
  DATA_TYPE4 out1 = 0;
  DATA_TYPE4 out2 = 0;
  DATA_TYPE4 out3 = 0;
#endif

  const short out_h = out_hb % out_height;
#if STRIDE == 1
  const short in_width0 = out_w_blk - padding_left;
  const short in_width1 = in_width0 + out_w_blks;
  const short in_width2 = in_width1 + out_w_blks;
  const short in_width3 = in_width2 + out_w_blks;
  const short height_idx = out_h - padding_top;
#elif STRIDE == 2
  int in_width0 = (out_w_blk << 1) - padding_left;
  int in_width1 = ((out_w_blk + out_w_blks) << 1) - padding_left;
  int in_width2 = ((out_w_blk + (out_w_blks << 1)) << 1) - padding_left;
  int in_width3 = ((out_w_blk + (out_w_blks << 1) + out_w_blks) << 1) - padding_left;
  const int height_idx = (out_h << 1) - padding_top;
#else
  const short in_width_stride = mul24(out_w_blks, STRIDE);
  const short in_width0 = mad24(out_w_blk, STRIDE, -padding_left);
  const short in_width1 = in_width0 + in_width_stride;
  const short in_width2 = in_width1 + in_width_stride;
  const short in_width3 = in_width2 + in_width_stride;
  const short height_idx = mad24(out_h, STRIDE, -padding_top);
#endif

  const short batch_idx = mul24((out_hb / out_height), in_height);
  const short rounded_in_ch_x_filter_width = mul24(rounded_in_ch, filter_width);

  const short in_idx = mul24(in_ch_blk, in_width);
  short filter_idx = 0;
  short in_hb_idx = height_idx;
  for (short filter_h_idx = 0; filter_h_idx < filter_height; ++filter_h_idx) {
    short in_hb = select(in_hb_idx + batch_idx,
                         -1,
                         (in_hb_idx < 0 || in_hb_idx >= in_height));
    short in_w_idx = 0;
    for (short filter_w_idx = 0; filter_w_idx < filter_width; ++filter_w_idx) {
      short in_w;
      DATA_TYPE4 in0, in1, in2, in3;
#define READ_INPUT(i)                                   \
      in_w = in_w_idx + in_width##i; \
      in_w = select(in_idx + in_w,                      \
                    -1,                                 \
                    (in_w < 0 || in_w >= in_width));    \
      in##i = READ_IMAGET(input, SAMPLER, (int2)(in_w, in_hb));

      READ_INPUT(0);
      READ_INPUT(1);
      READ_INPUT(2);
      READ_INPUT(3);

#undef READ_INPUT

      DATA_TYPE4 weights = READ_IMAGET(filter, SAMPLER,
                                       (int2)(filter_idx, in_ch_blk));

      out0 = mad(in0, weights, out0);
      out1 = mad(in1, weights, out1);
      out2 = mad(in2, weights, out2);
      out3 = mad(in3, weights, out3);
      ++filter_idx;
      in_w_idx += dilation_w;
    }
    in_hb_idx += dilation_h;
  }

#if defined(USE_RELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
  out0 = do_activation(out0, relux_max_limit);
  out1 = do_activation(out1, relux_max_limit);
  out2 = do_activation(out2, relux_max_limit);
  out3 = do_activation(out3, relux_max_limit);
#endif

  const short out_x_base = mul24(out_ch_blk, out_width);
  short w = out_w_blk;
  WRITE_IMAGET(output, (int2)(out_x_base + w, out_hb), out0);

  w += out_w_blks;
  if (w >= out_width) return;
  WRITE_IMAGET(output, (int2)(out_x_base + w, out_hb), out1);

  w += out_w_blks;
  if (w >= out_width) return;
  WRITE_IMAGET(output, (int2)(out_x_base + w, out_hb), out2);

  w += out_w_blks;
  if (w >= out_width) return;
  WRITE_IMAGET(output, (int2)(out_x_base + w, out_hb), out3);
}

__kernel void depthwise_conv2d_s1(KERNEL_ERROR_PARAMS
                                  GLOBAL_WORK_GROUP_SIZE_DIM3
                                  __read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
                                  __read_only image2d_t filter, /* cout%4 * kh * kw * m, cin/4 */
#ifdef BIAS
    __read_only image2d_t bias, /* cout%4 * cout/4 */
#endif
                                  __write_only image2d_t output,
                                  __private const DATA_TYPE relux_max_limit,
                                  __private const short in_height,
                                  __private const short in_width,
                                  __private const short in_ch_blks,
                                  __private const short out_height,
                                  __private const short out_width,
                                  __private const short filter_height,
                                  __private const short filter_width,
                                  __private const short padding_top,
                                  __private const short padding_left) {
  const short out_ch_blk = get_global_id(0);
  const short out_w_blk = get_global_id(1) << 2;
  const short out_hb = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (out_ch_blk >= global_size_dim0 || get_global_id(1) >= global_size_dim1
      || out_hb >= global_size_dim2) {
    return;
  }
#endif

  const short rounded_in_ch = in_ch_blks << 2;
  const short in_ch_blk = out_ch_blk; // multiplier = 1

#ifdef BIAS
  DATA_TYPE4 out0 =
     READ_IMAGET(bias, SAMPLER, (int2)(out_ch_blk, 0));
  DATA_TYPE4 out1 = out0;
  DATA_TYPE4 out2 = out0;
  DATA_TYPE4 out3 = out0;
#else
  DATA_TYPE4 out0 = 0;
  DATA_TYPE4 out1 = 0;
  DATA_TYPE4 out2 = 0;
  DATA_TYPE4 out3 = 0;
#endif

  const short out_h = out_hb % out_height;
  const short in_width0 = out_w_blk - padding_left;
  const short in_width1 = in_width0 + 1;
  const short in_width2 = in_width1 + 1;
  const short in_width3 = in_width2 + 1;
  const short height_idx = out_h - padding_top;

  const short batch_idx = mul24((out_hb / out_height), in_height);
  const short rounded_in_ch_x_filter_width = mul24(rounded_in_ch, filter_width);

  const short in_idx = mul24(in_ch_blk, in_width);
  short filter_idx = 0;
  short in_hb_idx = height_idx;

  const short in_w_idx0 = select(in_idx + in_width0,
                                 -1,
                                 (in_width0 < 0 || in_width0 >= in_width));
  const short in_w_idx1 = select(in_idx + in_width1,
                                 -1,
                                 (in_width1 < 0 || in_width1 >= in_width));
  const short in_w_idx2 = select(in_idx + in_width2,
                                 -1,
                                 (in_width2 < 0 || in_width2 >= in_width));

  short in_w;
  DATA_TYPE4 in0, in1, in2, in3;
  for (short filter_h_idx = 0; filter_h_idx < filter_height; ++filter_h_idx) {
    short in_hb = select(in_hb_idx + batch_idx,
                         -1,
                         (in_hb_idx < 0 || in_hb_idx >= in_height));
    in1 = READ_IMAGET(input, SAMPLER, (int2)(in_w_idx0, in_hb));
    in2 = READ_IMAGET(input, SAMPLER, (int2)(in_w_idx1, in_hb));
    in3 = READ_IMAGET(input, SAMPLER, (int2)(in_w_idx2, in_hb));

    for (short filter_w_idx = 0; filter_w_idx < filter_width; ++filter_w_idx) {
      in0 = in1;
      in1 = in2;
      in2 = in3;

      in_w = in_width3 + filter_w_idx;
      in_w = select(in_idx + in_w,
                    -1,
                    (in_w < 0 || in_w >= in_width));
      in3 = READ_IMAGET(input, SAMPLER, (int2)(in_w, in_hb));

      DATA_TYPE4 weights = READ_IMAGET(filter, SAMPLER,
                                       (int2)(filter_idx, in_ch_blk));

      out0 = mad(in0, weights, out0);
      out1 = mad(in1, weights, out1);
      out2 = mad(in2, weights, out2);
      out3 = mad(in3, weights, out3);
      ++filter_idx;
    }
    in_hb_idx += 1;
  }

#if defined(USE_RELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
  out0 = do_activation(out0, relux_max_limit);
  out1 = do_activation(out1, relux_max_limit);
  out2 = do_activation(out2, relux_max_limit);
  out3 = do_activation(out3, relux_max_limit);
#endif

  const short out_x_base = mul24(out_ch_blk, out_width);
  short w = out_w_blk;
  WRITE_IMAGET(output, (int2)(out_x_base + w, out_hb), out0);

  w += 1;
  if (w >= out_width) return;
  WRITE_IMAGET(output, (int2)(out_x_base + w, out_hb), out1);

  w += 1;
  if (w >= out_width) return;
  WRITE_IMAGET(output, (int2)(out_x_base + w, out_hb), out2);

  w += 1;
  if (w >= out_width) return;
  WRITE_IMAGET(output, (int2)(out_x_base + w, out_hb), out3);
}
