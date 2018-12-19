#include <common.h>

__kernel void deconv_2d(OUT_OF_RANGE_PARAMS
                        GLOBAL_WORK_GROUP_SIZE_DIM3
                        __read_only image2d_t input,
                        __read_only image2d_t weights,
#ifdef BIAS
                        __read_only image2d_t bias,
#endif
                        __write_only image2d_t output,
                        __private const float relux_max_limit,
                        __private const float leakyrelu_coefficient,
                        __private const int in_height,
                        __private const int in_width,
                        __private const int in_channels,
                        __private const int out_height,
                        __private const int out_width,
                        __private const int out_channel,
                        __private const int stride_h,
                        __private const int stride_w,
                        __private const float stride_h_r,
                        __private const float stride_w_r,
                        __private const int align_h,
                        __private const int align_w,
                        __private const int padding_h,
                        __private const int padding_w,
                        __private const int kernel_h,
                        __private const int kernel_w,
                        __private const int kernel_size,
                        __private const int in_channel_blocks,
                        __private const int out_channel_blocks)
{
  const int c = get_global_id(0);
  const int w_id = get_global_id(1);
  const int hb = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (c >= global_size_dim0 || w_id >= global_size_dim1
      || hb >= global_size_dim2) {
    return;
  }
#endif

#ifdef BIAS
  DATA_TYPE4 out0 =
     READ_IMAGET(bias, SAMPLER, (int2)(c, 0));
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

  const int n_stride = mad(w_id, stride_w_r, 0);
  const int mod_stride = w_id - mul24(n_stride, stride_w);
  const int w = mad24(mul24(n_stride, 5), stride_w, mod_stride);
  const int b = hb / out_height;
  const int h = hb - mul24(b, out_height);
  if (w < out_width) {
    int start_x = floor((float) (w + align_w) * stride_w_r);
    int start_y = (h + align_h) * stride_h_r;
    start_y = max(0, start_y);

    int f_start_x = mad24(start_x, stride_w, padding_w) - w;
    int f_start_y = mad24(start_y, stride_h, padding_h) - h;
    f_start_x = kernel_w - 1 - f_start_x;
    f_start_y = kernel_h - 1 - f_start_y;

    int2 in_pos;
    int f_pos_x0, f_pos_x1, f_pos_x2, f_pos_x3, f_pos_y;
    DATA_TYPE4 in0, in1, in2, in3, in4;
    DATA_TYPE4 weight0, weight1, weight2, weight3;
    int idx_w0, idx_w1, idx_w2, idx_w3, idx_w4;
    int index_x, index_y;
    for (int ic = 0; ic < in_channel_blocks; ++ic) {
      f_pos_x0 = mul24(ic, 4);
      f_pos_x1 = f_pos_x0 + 1;
      f_pos_x2 = f_pos_x0 + 2;
      f_pos_x3 = f_pos_x0 + 3;
      for (int f_y = f_start_y, idx_h = start_y ; f_y >= 0; f_y -= stride_h, ++idx_h) {
        index_y = mad24(b, in_height, idx_h);
        in_pos.y = select(index_y, -1, idx_h < 0 || idx_h >= in_height);
        for (int f_x = f_start_x, idx_w = start_x; f_x >= 0; f_x -= stride_w, ++idx_w) {
          f_pos_y = mad24(f_y, kernel_w, f_x);
          f_pos_y = mad24(c, kernel_size, f_pos_y);
          weight0 = READ_IMAGET(weights, SAMPLER, (int2)(f_pos_x0, f_pos_y));
          weight1 = READ_IMAGET(weights, SAMPLER, (int2)(f_pos_x1, f_pos_y));
          weight2 = READ_IMAGET(weights, SAMPLER, (int2)(f_pos_x2, f_pos_y));
          weight3 = READ_IMAGET(weights, SAMPLER, (int2)(f_pos_x3, f_pos_y));

          idx_w0 = idx_w;
          idx_w1 = idx_w + 1;
          idx_w2 = idx_w + 2;
          idx_w3 = idx_w + 3;
          idx_w4 = idx_w + 4;

#define READ_INPUT(i)                                                         \
          index_x = mad24(ic, in_width, idx_w##i);                            \
          in_pos.x =                                                          \
            select(index_x, -1, idx_w##i < 0 || idx_w##i >= in_width);        \
          in##i = READ_IMAGET(input, SAMPLER, in_pos);

          READ_INPUT(0);
          READ_INPUT(1);
          READ_INPUT(2);
          READ_INPUT(3);
          READ_INPUT(4);
#undef READ_INPUT

#define CALC_OUTPUT(i)                                                        \
          out##i = mad(in##i.x, weight0, out##i);                             \
          out##i = mad(in##i.y, weight1, out##i);                             \
          out##i = mad(in##i.z, weight2, out##i);                             \
          out##i = mad(in##i.w, weight3, out##i);

          CALC_OUTPUT(0);
          CALC_OUTPUT(1);
          CALC_OUTPUT(2);
          CALC_OUTPUT(3);
          CALC_OUTPUT(4);
#undef CALC_OUTPUT
        }
      }
    }

#if  defined(USE_RELU) || defined(USE_LEAKYRELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
    out0 = do_activation(out0, relux_max_limit, leakyrelu_coefficient);
    out1 = do_activation(out1, relux_max_limit, leakyrelu_coefficient);
    out2 = do_activation(out2, relux_max_limit, leakyrelu_coefficient);
    out3 = do_activation(out3, relux_max_limit, leakyrelu_coefficient);
    out4 = do_activation(out4, relux_max_limit, leakyrelu_coefficient);
#endif

    int2 out_pos;
    out_pos.y = hb;

    int ow = w;
    if (ow >= out_width) return;
    out_pos.x = mad24(c, out_width, ow);
    WRITE_IMAGET(output, out_pos, out0);

    ow += stride_w;
    if (ow >= out_width) return;
    out_pos.x += stride_w;
    WRITE_IMAGET(output, out_pos, out1);

    ow += stride_w;
    if (ow >= out_width) return;
    out_pos.x += stride_w;
    WRITE_IMAGET(output, out_pos, out2);

    ow += stride_w;
    if (ow >= out_width) return;
    out_pos.x += stride_w;
    WRITE_IMAGET(output, out_pos, out3);

    ow += stride_w;
    if (ow >= out_width) return;
    out_pos.x += stride_w;
    WRITE_IMAGET(output, out_pos, out4);
  }
}

