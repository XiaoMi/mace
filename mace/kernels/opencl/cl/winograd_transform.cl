#include <common.h>

__kernel void winograd_transform_2x2(GLOBAL_WORK_GROUP_SIZE_DIM2
                                     __read_only image2d_t input,
                                     __write_only image2d_t output,
                                     __private const int in_height,
                                     __private const int in_width,
                                     __private const int in_channel,
                                     __private const int round_hw,
                                     __private const int round_w,
                                     __private const int padding_top,
                                     __private const int padding_left) {
  int out_width_idx = get_global_id(0);
  int chan_blk_idx = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (out_width_idx >= global_size_dim0 || chan_blk_idx >= global_size_dim1) {
    return;
  }
  const int chan_blk_size = global_size_dim1;
#else
  const int chan_blk_size = get_global_size(1);
#endif

  const int batch_idx = out_width_idx / round_hw;
  const int t_idx = out_width_idx % round_hw;
  const int height_idx = ((t_idx / round_w) << 1) - padding_top;
  const int width_idx = ((t_idx % round_w) << 1) - padding_left;

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

__kernel void winograd_inverse_transform_2x2(GLOBAL_WORK_GROUP_SIZE_DIM2
                                             __read_only image2d_t input,
#ifdef BIAS
                                             __read_only image2d_t bias, /* cout%4 * cout/4 */
#endif
                                             __write_only image2d_t output,
                                             __private const int out_height,
                                             __private const int out_width,
                                             __private const int round_hw,
                                             __private const int round_w,
                                             __private const float relux_max_limit) {
  const int width_idx = get_global_id(0);
  const int height_idx = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (width_idx >= global_size_dim0 || height_idx >= global_size_dim1) {
    return;
  }
  const int out_channel = global_size_dim1;
#else
  const int out_channel = get_global_size(1);
#endif

  int width = width_idx;
  int height = height_idx;

  const int batch = width_idx / round_hw;
  int t = width_idx % round_hw;
  const int out_height_idx = (t / round_w) << 1;
  const int out_width_idx = (t % round_w) << 1;
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
