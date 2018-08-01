#include <common.h>

__kernel void conv_2d_3x3(KERNEL_ERROR_PARAMS
                          GLOBAL_WORK_GROUP_SIZE_DIM3
                          __read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
                          __read_only image2d_t filter, /* cout%4 * cin , kh * kw * cout/4 */
#ifdef BIAS
                          __read_only image2d_t bias, /* cout%4 * cout/4 */
#endif
                          __write_only image2d_t output,
                          __private const float relux_max_limit,
                          __private const int in_height,
                          __private const int in_width,
                          __private const int in_ch_blks,
                          __private const int out_height,
                          __private const int out_width,
                          __private const int stride,
                          __private const int padding_top,
                          __private const int padding_left,
                          __private const int dilation_h,
                          __private const int dilation_w) {
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
  DATA_TYPE4 out0 =
     READ_IMAGET(bias, SAMPLER, (int2)(out_ch_blk, 0));
  DATA_TYPE4 out1 = out0;
  DATA_TYPE4 out2 = out0;
  DATA_TYPE4 out3 = out0;
  DATA_TYPE4 out4 = out0;
#else
  DATA_TYPE4 out0 = 0;
  DATA_TYPE4 out1 = 0;
  DATA_TYPE4 out2 = 0;
  DATA_TYPE4 out3 = 0;
  DATA_TYPE4 out4 = 0;
#endif

  int in_width_stride = mul24(out_w_blks, stride);
  int in_width0 = mad24(out_w_blk, stride, -padding_left);
  int in_width1 = in_width0 + in_width_stride;
  int in_width2 = in_width1 + in_width_stride;
  int in_width3 = in_width2 + in_width_stride;
  int in_width4 = in_width3 + in_width_stride;
  const int height_start = mad24((out_hb % out_height), stride, -padding_top);
  int in_height_gap = select(
      0,
      (-height_start + dilation_h - 1) / dilation_h,
      height_start < 0);
  int in_height_start = mad24(in_height_gap, dilation_h, height_start);
  int in_height_end = min(mad24(3, dilation_h, height_start),
                          in_height);

  const int batch_idx = mul24((out_hb / out_height), in_height);
  const int filter_y_idx_start = mul24(out_ch_blk, 9) + mul24(in_height_gap, 3);

  DATA_TYPE4 in0, in1, in2, in3, in4;
  DATA_TYPE4 weights0, weights1, weights2, weights3;
  for (short in_ch_blk = 0; in_ch_blk < in_ch_blks; ++in_ch_blk) {
    const int in_idx = mul24(in_ch_blk, in_width);
    int filter_x_idx = in_ch_blk << 2;
    int filter_y_idx = filter_y_idx_start;
    for (int hb_idx = in_height_start; hb_idx < in_height_end; hb_idx += dilation_h) {
      int in_hb_value = hb_idx + batch_idx;
      int in_width_idx = 0;
      for (short width_idx = 0; width_idx < 3; ++width_idx) {
        int in_width_value;
#define READ_INPUT(i)                                                                \
        in_width_value = in_width##i + in_width_idx;                                 \
        in_width_value = select(in_idx + in_width_value,                             \
                                -1,                                                  \
                                (in_width_value < 0 || in_width_value >= in_width)); \
        in##i = READ_IMAGET(input, SAMPLER, (int2)(in_width_value, in_hb_value));

        READ_INPUT(0);
        READ_INPUT(1);
        READ_INPUT(2);
        READ_INPUT(3);
        READ_INPUT(4);

#undef READ_INPUT

        // int filter_idx = (hb_idx * 3 + width_idx) * in_ch + (in_ch_blk << 2);
        weights0 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_idx + 0, filter_y_idx));
        weights1 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_idx + 1, filter_y_idx));
        weights2 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_idx + 2, filter_y_idx));
        weights3 = READ_IMAGET(filter, SAMPLER, (int2)(filter_x_idx + 3, filter_y_idx));

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

        out4 = mad(in4.x, weights0, out4);
        out4 = mad(in4.y, weights1, out4);
        out4 = mad(in4.z, weights2, out4);
        out4 = mad(in4.w, weights3, out4);

        in_width_idx += dilation_w;
        filter_y_idx += 1;
      }
    }
  }

#if defined(USE_RELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
  out0 = do_activation(out0, relux_max_limit);
  out1 = do_activation(out1, relux_max_limit);
  out2 = do_activation(out2, relux_max_limit);
  out3 = do_activation(out3, relux_max_limit);
  out4 = do_activation(out4, relux_max_limit);
#endif

  const int out_x_base = mul24(out_ch_blk, out_width);
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
