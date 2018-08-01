#include <common.h>

// output = weight * input + bias
__kernel void fully_connected(KERNEL_ERROR_PARAMS
                              GLOBAL_WORK_GROUP_SIZE_DIM2
                              __read_only image2d_t input,
                              __read_only image2d_t weight,
#ifdef BIAS
    __read_only image2d_t bias,
#endif
                              __write_only image2d_t output,
                              __private const int input_height,
                              __private const int input_width,
                              __private const int input_channel,
                              __private const float relux_max_limit) {
  const int batch_idx = get_global_id(0);
  const int out_blk_idx = get_global_id(1);
  const int input_chan_blk = (input_channel + 3) >> 2;

#ifndef NON_UNIFORM_WORK_GROUP
  if (batch_idx >= global_size_dim0 || out_blk_idx >= global_size_dim1) return;
#endif

  float4 input_value;
  float4 w0, w1, w2, w3;

#ifdef BIAS
  DATA_TYPE4 result = READ_IMAGET(bias, SAMPLER, (int2)(out_blk_idx, 0));
#else
  DATA_TYPE4 result = (DATA_TYPE4)(0, 0, 0, 0);
#endif

  int2 input_coord = (int2)(0, mul24(batch_idx, input_height));
  int weight_x = 0;
  for (short h_idx = 0; h_idx < input_height; ++h_idx) {
    for (short w_idx = 0; w_idx < input_width; ++w_idx) {
      input_coord.x = w_idx;
      weight_x = (h_idx * input_width + w_idx) * input_channel;
#pragma unroll
      for (short chan_idx = 0; chan_idx < input_chan_blk; ++chan_idx) {
        input_value = READ_IMAGET(input, SAMPLER, input_coord);

        w0 = READ_IMAGET(weight, SAMPLER, (int2)(weight_x++, out_blk_idx));
        w1 = READ_IMAGET(weight, SAMPLER, (int2)(weight_x++, out_blk_idx));
        w2 = READ_IMAGET(weight, SAMPLER, (int2)(weight_x++, out_blk_idx));
        w3 = READ_IMAGET(weight, SAMPLER, (int2)(weight_x++, out_blk_idx));

        result = mad(input_value.x, w0, result);
        result = mad(input_value.y, w1, result);
        result = mad(input_value.z, w2, result);
        result = mad(input_value.w, w3, result);

        input_coord.x += input_width;
      }
    }
    input_coord.y++;
  }

#if defined(USE_RELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
  result = do_activation(result, relux_max_limit);
#endif

  WRITE_IMAGET(output, (int2)(out_blk_idx, batch_idx), result);
}

// output = weight * input + bias
__kernel void fully_connected_width(KERNEL_ERROR_PARAMS
                                    GLOBAL_WORK_GROUP_SIZE_DIM3
                                    __read_only image2d_t input,
                                    __read_only image2d_t weight,
#ifdef BIAS
    __read_only image2d_t bias,
#endif
                                    __write_only image2d_t output,
                                    __local float *intermediate_output,
                                    __private const int input_height,
                                    __private const int input_width,
                                    __private const int in_chan_blks,
                                    __private const int out_blks,
                                    __private const float relux_max_limit) {
  const int inter_out_idx = get_global_id(0);
  const int width_blk_idx = get_global_id(1);
  const int width_blk_count = global_size_dim1;
  const int batch_out_blk_idx = get_global_id(2);

  const int batch_idx = batch_out_blk_idx / out_blks;
  const int out_blk_idx = batch_out_blk_idx % out_blks;

  const short in_outer_size = mul24(input_width, in_chan_blks);
  const short weight_y = mad24(out_blk_idx, 4, inter_out_idx);

  int2 input_coord, weight_coord;
  DATA_TYPE4 in, w;
  DATA_TYPE sum = 0;

  input_coord = (int2)(0, mul24(batch_idx, input_height));

  for (int h_idx = 0; h_idx < input_height; ++h_idx) {
    int weight_x_base = mul24(h_idx, in_outer_size);
    for (int w_idx = width_blk_idx; w_idx < input_width;
         w_idx += width_blk_count) {
      int weight_x = mad24(w_idx, in_chan_blks, weight_x_base);
      weight_coord = (int2)(weight_x, weight_y);
      input_coord.x = w_idx;
#pragma unroll
      for (int chan_idx = 0; chan_idx < in_chan_blks; ++chan_idx) {
        in = READ_IMAGET(input, SAMPLER, input_coord);

        w = READ_IMAGET(weight, SAMPLER, weight_coord);

        sum += dot(in, w);

        input_coord.x += input_width;
        weight_coord.x += 1;
      }
    }
    input_coord.y++;
  }

  const short inter_out_offset = mad24((short)get_local_id(1), (short)4,
                                       (short)get_local_id(0));
  const short local_width_blk_size = (short)get_local_size(1);
  const short local_size = mul24((short)get_local_size(0),
                                 local_width_blk_size);
  short inter_idx = mad24((short)get_local_id(2), local_size, inter_out_offset);
  intermediate_output[inter_idx] = sum;

#ifdef NON_QUALCOMM_ADRENO
  barrier(CLK_LOCAL_MEM_FENCE);
#endif

#ifndef NON_UNIFORM_WORK_GROUP
  if (batch_out_blk_idx >= global_size_dim2) {
    return;
  }
#endif

  if (inter_out_offset == 0) {
#ifdef BIAS
    DATA_TYPE4 result = READ_IMAGET(bias, SAMPLER, (int2)(out_blk_idx, 0));
#else
    DATA_TYPE4 result = (DATA_TYPE4)(0, 0, 0, 0);
#endif

    for (short i = 0; i < local_width_blk_size; ++i) {
      result += vload4(0, intermediate_output+inter_idx);
      inter_idx += 4;
    }

#if defined(USE_RELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
    result = do_activation(result, relux_max_limit);
#endif

    WRITE_IMAGET(output, (int2)(out_blk_idx, batch_idx), result);
  }
}
