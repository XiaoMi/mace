#include <common.h>

__kernel void winograd_transform_2x2(KERNEL_ERROR_PARAMS
                                     GLOBAL_WORK_GROUP_SIZE_DIM2
                                     __read_only image2d_t input,
                                     __write_only image2d_t output,
                                     __private const int in_height,
                                     __private const int in_width,
                                     __private const int in_channel,
                                     __private const int round_hw,
                                     __private const float round_hw_r,
                                     __private const int round_w,
                                     __private const float round_w_r,
                                     __private const int padding_top,
                                     __private const int padding_left) {
  int out_width_idx = get_global_id(0);
  int chan_blk_idx = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (out_width_idx >= global_size_dim0 || chan_blk_idx >= global_size_dim1) {
    return;
  }
#endif
  const int chan_blk_size = global_size_dim1;

  const int batch_idx = out_width_idx * round_hw_r;
  const int t_idx = mad24(batch_idx, -round_hw, out_width_idx);
  const int n_round_w = t_idx * round_w_r;
  const int mod_round_w = mad24(n_round_w, -round_w, t_idx);
  const int height_idx = (n_round_w << 1) - padding_top;
  const int width_idx = (mod_round_w << 1) - padding_left;

  const int nh_idx = mad24(batch_idx, in_height, height_idx);
  const int wc_idx = mad24(chan_blk_idx, in_width, width_idx);

  DATA_TYPE4 input0[4];
  DATA_TYPE4 input1[4];
  DATA_TYPE4 input2[4];
  DATA_TYPE4 input3[4];

  DATA_TYPE4 tv0[4];
  DATA_TYPE4 tv1[4];
  DATA_TYPE4 tv2[4];
  DATA_TYPE4 tv3[4];

  int y = select(nh_idx, -1, height_idx < 0 || height_idx >= in_height);
#pragma unroll
  for (short i = 0; i < 4; ++i) {
    int x = width_idx + i;
    x = select(wc_idx + i, -1, x < 0 || x >= in_width);
    input0[i] = READ_IMAGET(input, SAMPLER, (int2)(x, y));
  }
  y = select(nh_idx + 1, -1, height_idx + 1 < 0 || height_idx + 1 >= in_height);
#pragma unroll
  for (short i = 0; i < 4; ++i) {
    int x = width_idx + i;
    x = select(wc_idx + i, -1, x < 0 || x >= in_width);
    input1[i] = READ_IMAGET(input, SAMPLER, (int2)(x, y));
  }
  y = select(nh_idx + 2, -1, height_idx + 2 < 0 || height_idx + 2 >= in_height);
#pragma unroll
  for (short i = 0; i < 4; ++i) {
    int x = width_idx + i;
    x = select(wc_idx + i, -1, x < 0 || x >= in_width);
    input2[i] = READ_IMAGET(input, SAMPLER, (int2)(x, y));
  }
  y = select(nh_idx + 3, -1, height_idx + 3 < 0 || height_idx + 3 >= in_height);
#pragma unroll
  for (short i = 0; i < 4; ++i) {
    int x = width_idx + i;
    x = select(wc_idx + i, -1, x < 0 || x >= in_width);
    input3[i] = READ_IMAGET(input, SAMPLER, (int2)(x, y));
  }

#pragma unroll
  for (short i = 0; i < 4; ++i) {
    tv0[i] = input0[i] - input2[i];
    tv1[i] = input1[i] + input2[i];
    tv2[i] = input2[i] - input1[i];
    tv3[i] = input1[i] - input3[i];
  }
  input0[0] = tv0[0] - tv0[2];
  input0[1] = tv0[1] + tv0[2];
  input0[2] = tv0[2] - tv0[1];
  input0[3] = tv0[1] - tv0[3];
  input1[0] = tv1[0] - tv1[2];
  input1[1] = tv1[1] + tv1[2];
  input1[2] = tv1[2] - tv1[1];
  input1[3] = tv1[1] - tv1[3];
  input2[0] = tv2[0] - tv2[2];
  input2[1] = tv2[1] + tv2[2];
  input2[2] = tv2[2] - tv2[1];
  input2[3] = tv2[1] - tv2[3];
  input3[0] = tv3[0] - tv3[2];
  input3[1] = tv3[1] + tv3[2];
  input3[2] = tv3[2] - tv3[1];
  input3[3] = tv3[1] - tv3[3];

#pragma unroll
  for (short i = 0; i < 4; ++i) {
    WRITE_IMAGET(output, (int2)(out_width_idx, chan_blk_idx), input0[i]);
    chan_blk_idx += chan_blk_size;
  }
#pragma unroll
  for (short i = 0; i < 4; ++i) {
    WRITE_IMAGET(output, (int2)(out_width_idx, chan_blk_idx), input1[i]);
    chan_blk_idx += chan_blk_size;
  }
#pragma unroll
  for (short i = 0; i < 4; ++i) {
    WRITE_IMAGET(output, (int2)(out_width_idx, chan_blk_idx), input2[i]);
    chan_blk_idx += chan_blk_size;
  }
#pragma unroll
  for (short i = 0; i < 4; ++i) {
    WRITE_IMAGET(output, (int2)(out_width_idx, chan_blk_idx), input3[i]);
    chan_blk_idx += chan_blk_size;
  }
}

__kernel void winograd_inverse_transform_2x2(KERNEL_ERROR_PARAMS
                                             GLOBAL_WORK_GROUP_SIZE_DIM2
                                             __read_only image2d_t input,
#ifdef BIAS
                                             __read_only image2d_t bias, /* cout%4 * cout/4 */
#endif
                                             __write_only image2d_t output,
                                             __private const int out_height,
                                             __private const int out_width,
                                             __private const int round_hw,
                                             __private const float round_hw_r,
                                             __private const int round_w,
                                             __private const float round_w_r,
                                             __private const float relux_max_limit) {
  const int width_idx = get_global_id(0);
  const int height_idx = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (width_idx >= global_size_dim0 || height_idx >= global_size_dim1) {
    return;
  }
#endif
  const int out_channel = global_size_dim1;

  int width = width_idx;
  int height = height_idx;

  const int batch = width_idx * round_hw_r;
  int t = mad24(batch, -round_hw, width_idx);
  const int n_round_w = t * round_w_r;
  const int mod_round_w = mad24(n_round_w, -round_w, t);
  const int out_height_idx = n_round_w << 1;
  const int out_width_idx = mod_round_w << 1;
  const int out_chan_idx = height_idx;
  const int coord_x = mad24(out_chan_idx, out_width, out_width_idx);
  const int coord_y = mad24(batch, out_height, out_height_idx);

#ifdef BIAS
  DATA_TYPE4 bias_value =
     READ_IMAGET(bias, SAMPLER, (int2)(out_chan_idx, 0));
#endif

  DATA_TYPE4 in0[4], in1[4], in2[4], in3[4];

#pragma unroll
  for (short i = 0; i < 4; ++i) {
    in0[i] = READ_IMAGET(input, SAMPLER, (int2)(width, height));
    height += out_channel;
  }
#pragma unroll
  for (short i = 0; i < 4; ++i) {
    in1[i] = READ_IMAGET(input, SAMPLER, (int2)(width_idx, height));
    height += out_channel;
  }
#pragma unroll
  for (short i = 0; i < 4; ++i) {
    in2[i] = READ_IMAGET(input, SAMPLER, (int2)(width_idx, height));
    height += out_channel;
  }
#pragma unroll
  for (short i = 0; i < 4; ++i) {
    in3[i] = READ_IMAGET(input, SAMPLER, (int2)(width_idx, height));
    height += out_channel;
  }

  in0[0] = in0[0] + in1[0] + in2[0];
  in0[1] = in0[1] + in1[1] + in2[1];
  in0[2] = in0[2] + in1[2] + in2[2];
  in0[3] = in0[3] + in1[3] + in2[3];

  in0[0] = in0[0] + in0[1] + in0[2];
  in0[1] = in0[1] - in0[2] - in0[3];

  in1[0] = in1[0] - in2[0] - in3[0];
  in1[1] = in1[1] - in2[1] - in3[1];
  in1[2] = in1[2] - in2[2] - in3[2];
  in1[3] = in1[3] - in2[3] - in3[3];

  in1[0] = in1[0] + in1[1] + in1[2];
  in1[1] = in1[1] - in1[2] - in1[3];

#ifdef BIAS
  in0[0] += bias_value;
  in0[1] += bias_value;
  in1[0] += bias_value;
  in1[1] += bias_value;
#endif


#if defined(USE_RELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
  in0[0] = do_activation(in0[0], relux_max_limit);
  in0[1] = do_activation(in0[1], relux_max_limit);
  in1[0] = do_activation(in1[0], relux_max_limit);
  in1[1] = do_activation(in1[1], relux_max_limit);
#endif

  WRITE_IMAGET(output, (int2)(coord_x, coord_y), in0[0]);

  t = 0;
  if (out_width_idx + 1 < out_width) {
    WRITE_IMAGET(output, (int2)(coord_x + 1, coord_y), in0[1]);
    t += 1;
  }
  if (out_height_idx + 1 < out_height) {
    WRITE_IMAGET(output, (int2)(coord_x, coord_y + 1), in1[0]);
    t += 1;
  }
  if (t == 2) {
    WRITE_IMAGET(output, (int2)(coord_x + 1, coord_y + 1), in1[1]);
  }

}

__kernel void winograd_transform_4x4(KERNEL_ERROR_PARAMS
                                     GLOBAL_WORK_GROUP_SIZE_DIM2
                                     __read_only image2d_t input,
                                     __write_only image2d_t output,
                                     __private const int in_height,
                                     __private const int in_width,
                                     __private const int in_channel,
                                     __private const int round_hw,
                                     __private const float round_hw_r,
                                     __private const int round_w,
                                     __private const float round_w_r,
                                     __private const int padding_top,
                                     __private const int padding_left) {
  int out_width_idx = get_global_id(0);
  int chan_blk_idx = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (out_width_idx >= global_size_dim0 || chan_blk_idx >= global_size_dim1) {
    return;
  }
#endif
  const int chan_blk_size = global_size_dim1;

  const int batch_idx = out_width_idx * round_hw_r;
  const int t_idx = mad24(batch_idx, -round_hw, out_width_idx);
  const int n_round_w = t_idx * round_w_r;
  const int mod_round_w = mad24(n_round_w, -round_w, t_idx);
  const int height_idx = (n_round_w << 2) - padding_top;
  const int width_idx = (mod_round_w << 2) - padding_left;

  const int nh_idx = mad24(batch_idx, in_height, height_idx);
  const int wc_idx = mad24(chan_blk_idx, in_width, width_idx);

  DATA_TYPE4 in0[6], in1[6], in2[6], in3[6], in4[6], in5[6];
  DATA_TYPE4 tv0[6], tv1[6], tv2[6], tv3[6], tv4[6], tv5[6];

  int y = select(nh_idx, -1, height_idx < 0 || height_idx >= in_height);
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    int x = width_idx + i;
    x = select(wc_idx + i, -1, x < 0 || x >= in_width);
    in0[i] = READ_IMAGET(input, SAMPLER, (int2)(x, y));
  }
  y = select(nh_idx + 1, -1, height_idx + 1 < 0 || height_idx + 1 >= in_height);
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    int x = width_idx + i;
    x = select(wc_idx + i, -1, x < 0 || x >= in_width);
    in1[i] = READ_IMAGET(input, SAMPLER, (int2)(x, y));
  }
  y = select(nh_idx + 2, -1, height_idx + 2 < 0 || height_idx + 2 >= in_height);
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    int x = width_idx + i;
    x = select(wc_idx + i, -1, x < 0 || x >= in_width);
    in2[i] = READ_IMAGET(input, SAMPLER, (int2)(x, y));
  }
  y = select(nh_idx + 3, -1, height_idx + 3 < 0 || height_idx + 3 >= in_height);
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    int x = width_idx + i;
    x = select(wc_idx + i, -1, x < 0 || x >= in_width);
    in3[i] = READ_IMAGET(input, SAMPLER, (int2)(x, y));
  }
  y = select(nh_idx + 4, -1, height_idx + 4 < 0 || height_idx + 4 >= in_height);
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    int x = width_idx + i;
    x = select(wc_idx + i, -1, x < 0 || x >= in_width);
    in4[i] = READ_IMAGET(input, SAMPLER, (int2)(x, y));
  }
  y = select(nh_idx + 5, -1, height_idx + 5 < 0 || height_idx + 5 >= in_height);
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    int x = width_idx + i;
    x = select(wc_idx + i, -1, x < 0 || x >= in_width);
    in5[i] = READ_IMAGET(input, SAMPLER, (int2)(x, y));
  }
  DATA_TYPE4 tt0, tt1, tt2, tt3, tt4, tt5;
#define PROCESS_IN(i)         \
  tt0 = in2[i] - 4 * in0[i];  \
  tt1 = in3[i] - 4 * in1[i];  \
  tt2 = in4[i] - 4 * in2[i];  \
  tt3 = in5[i] - 4 * in3[i];  \
  tt4 = in3[i] - in1[i];      \
  tt4 = tt4 + tt4;            \
  tt5 = in4[i] - in2[i];      \
  tv0[i] = tt2 - tt0;         \
  tv1[i] = tt2 + tt1;         \
  tv2[i] = tt2 - tt1;         \
  tv3[i] = tt5 + tt4;         \
  tv4[i] = tt5 - tt4;         \
  tv5[i] = tt3 - tt1;

  PROCESS_IN(0);
  PROCESS_IN(1);
  PROCESS_IN(2);
  PROCESS_IN(3);
  PROCESS_IN(4);
  PROCESS_IN(5);

#undef PROCESS_IN

#define PROCESS_SND(i)            \
  tt0 = tv##i[2] - 4 * tv##i[0];  \
  tt1 = tv##i[3] - 4 * tv##i[1];  \
  tt2 = tv##i[4] - 4 * tv##i[2];  \
  tt3 = tv##i[5] - 4 * tv##i[3];  \
  tt4 = tv##i[3] - tv##i[1];      \
  tt4 = tt4 + tt4;                \
  tt5 = tv##i[4] - tv##i[2];      \
  in##i[0] = tt2 - tt0;           \
  in##i[1] = tt2 + tt1;           \
  in##i[2] = tt2 - tt1;           \
  in##i[3] = tt5 + tt4;           \
  in##i[4] = tt5 - tt4;           \
  in##i[5] = tt3 - tt1;

  PROCESS_SND(0);
  PROCESS_SND(1);
  PROCESS_SND(2);
  PROCESS_SND(3);
  PROCESS_SND(4);
  PROCESS_SND(5);

#undef PROCESS_SND

#pragma unroll
  for (short i = 0; i < 6; ++i) {
    WRITE_IMAGET(output, (int2)(out_width_idx, chan_blk_idx), in0[i]);
    chan_blk_idx += chan_blk_size;
  }
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    WRITE_IMAGET(output, (int2)(out_width_idx, chan_blk_idx), in1[i]);
    chan_blk_idx += chan_blk_size;
  }
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    WRITE_IMAGET(output, (int2)(out_width_idx, chan_blk_idx), in2[i]);
    chan_blk_idx += chan_blk_size;
  }
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    WRITE_IMAGET(output, (int2)(out_width_idx, chan_blk_idx), in3[i]);
    chan_blk_idx += chan_blk_size;
  }
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    WRITE_IMAGET(output, (int2)(out_width_idx, chan_blk_idx), in4[i]);
    chan_blk_idx += chan_blk_size;
  }
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    WRITE_IMAGET(output, (int2)(out_width_idx, chan_blk_idx), in5[i]);
    chan_blk_idx += chan_blk_size;
  }
}

__kernel void winograd_inverse_transform_4x4(KERNEL_ERROR_PARAMS
                                             GLOBAL_WORK_GROUP_SIZE_DIM2
                                             __read_only image2d_t input,
#ifdef BIAS
                                             __read_only image2d_t bias, /* cout%4 * cout/4 */
#endif
                                             __write_only image2d_t output,
                                             __private const int out_height,
                                             __private const int out_width,
                                             __private const int round_hw,
                                             __private const float round_hw_r,
                                             __private const int round_w,
                                             __private const float round_w_r,
                                             __private const float relux_max_limit) {
  const int width_idx = get_global_id(0);
  const int height_idx = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (width_idx >= global_size_dim0 || height_idx >= global_size_dim1) {
    return;
  }
#endif
  const int out_channel = global_size_dim1;

  const int batch = width_idx * round_hw_r;
  int h = mad24(batch, -round_hw, width_idx);
  int n_round_w = h * round_w_r;
  int mod_round_w = mad24(n_round_w, -round_w, h);
  const int out_height_idx = n_round_w << 2;
  const int out_width_idx = mod_round_w << 2;
  const int coord_x = mad24(height_idx, out_width, out_width_idx);
  const int coord_y = mad24(batch, out_height, out_height_idx);

#ifdef BIAS
  DATA_TYPE4 bias_value =
     READ_IMAGET(bias, SAMPLER, (int2)(height_idx, 0));
#endif

  DATA_TYPE4 out0[4], out1[4], out2[4], out3[4];
  DATA_TYPE4 in0[6], in1[6], in2[6], in3[6], in4[6], in5[6];
  h = height_idx;
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    in0[i] = READ_IMAGET(input, SAMPLER, (int2)(width_idx, h));
    h += out_channel;
  }
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    in1[i] = READ_IMAGET(input, SAMPLER, (int2)(width_idx, h));
    h += out_channel;
  }
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    in2[i] = READ_IMAGET(input, SAMPLER, (int2)(width_idx, h));
    h += out_channel;
  }
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    in3[i] = READ_IMAGET(input, SAMPLER, (int2)(width_idx, h));
    h += out_channel;
  }
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    in4[i] = READ_IMAGET(input, SAMPLER, (int2)(width_idx, h));
    h += out_channel;
  }
#pragma unroll
  for (short i = 0; i < 6; ++i) {
    in5[i] = READ_IMAGET(input, SAMPLER, (int2)(width_idx, h));
    h += out_channel;
  }

  DATA_TYPE4 tt0, tt1, tt2, tt3, d0, d5;
#define PROCESS_IN(i)             \
  d0 = in0[i];                    \
  d5 = in5[i];                    \
  tt0 = in1[i] + in2[i];          \
  tt1 = in1[i] - in2[i];          \
  tt2 = in3[i] + in4[i];          \
  tt3 = in3[i] - in4[i];          \
  tt3 = tt3 + tt3;                \
  in0[i] = d0 + tt0 + tt2;        \
  in1[i] = tt3 + tt1;             \
  in2[i] = tt2 * 4 + tt0;         \
  in3[i] = tt3 * 4 + tt1 + d5;

  PROCESS_IN(0);
  PROCESS_IN(1);
  PROCESS_IN(2);
  PROCESS_IN(3);
  PROCESS_IN(4);
  PROCESS_IN(5);

#undef PROCESS_IN

#define PROCESS_SND(i)                 \
  d0 = in##i[0];                       \
  d5 = in##i[5];                       \
  tt0 = in##i[1] + in##i[2];           \
  tt1 = in##i[1] - in##i[2];           \
  tt2 = in##i[3] + in##i[4];           \
  tt3 = in##i[3] - in##i[4];           \
  tt3 = tt3 + tt3;                     \
  out##i[0] = d0 + tt0 + tt2;          \
  out##i[1] = tt3 + tt1;               \
  out##i[2] = tt2 * 4 + tt0;           \
  out##i[3] = tt3 * 4 + tt1 + d5;

  PROCESS_SND(0);
  PROCESS_SND(1);
  PROCESS_SND(2);
  PROCESS_SND(3);
#undef PROCESS_SND

#ifdef BIAS
    out0[0] += bias_value;
    out0[1] += bias_value;
    out0[2] += bias_value;
    out0[3] += bias_value;
    out1[0] += bias_value;
    out1[1] += bias_value;
    out1[2] += bias_value;
    out1[3] += bias_value;
    out2[0] += bias_value;
    out2[1] += bias_value;
    out2[2] += bias_value;
    out2[3] += bias_value;
    out3[0] += bias_value;
    out3[1] += bias_value;
    out3[2] += bias_value;
    out3[3] += bias_value;
#endif

#if defined(USE_RELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
  out0[0] = do_activation(out0[0], relux_max_limit);
  out0[1] = do_activation(out0[1], relux_max_limit);
  out0[2] = do_activation(out0[2], relux_max_limit);
  out0[3] = do_activation(out0[3], relux_max_limit);
  out1[0] = do_activation(out1[0], relux_max_limit);
  out1[1] = do_activation(out1[1], relux_max_limit);
  out1[2] = do_activation(out1[2], relux_max_limit);
  out1[3] = do_activation(out1[3], relux_max_limit);
  out2[0] = do_activation(out2[0], relux_max_limit);
  out2[1] = do_activation(out2[1], relux_max_limit);
  out2[2] = do_activation(out2[2], relux_max_limit);
  out2[3] = do_activation(out2[3], relux_max_limit);
  out3[0] = do_activation(out3[0], relux_max_limit);
  out3[1] = do_activation(out3[1], relux_max_limit);
  out3[2] = do_activation(out3[2], relux_max_limit);
  out3[3] = do_activation(out3[3], relux_max_limit);
#endif

  const int num = min(4, out_width - out_width_idx);
  const int h_num = out_height - out_height_idx;
  if(h_num < 1) return;
#pragma unroll
  for (int i = 0; i < num; ++i) {
    WRITE_IMAGET(output, (int2)(coord_x + i, coord_y), out0[i]);
  }
  if(h_num < 2) return;
#pragma unroll
  for (int i = 0; i < num; ++i) {
    WRITE_IMAGET(output, (int2)(coord_x + i, coord_y + 1), out1[i]);
  }
  if(h_num < 3) return;
#pragma unroll
  for (int i = 0; i < num; ++i) {
    WRITE_IMAGET(output, (int2)(coord_x + i, coord_y + 2), out2[i]);
  }
  if(h_num < 4) return;
#pragma unroll
  for (int i = 0; i < num; ++i) {
    WRITE_IMAGET(output, (int2)(coord_x + i, coord_y + 3), out3[i]);
  }
}