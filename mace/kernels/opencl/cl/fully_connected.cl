#include <common.h>

// output = weight * input + bias
__kernel void fully_connected(__read_only image2d_t input,
                              __read_only image2d_t weight,
#ifdef BIAS
                              __read_only image2d_t bias,
#endif
                              __write_only image2d_t output,
                              __private const int input_height,
                              __private const int input_width,
                              __private const int input_channel,
                              __private const float relux_max_limit,
                              __private const float prelu_alpha) {
  const int batch_idx = get_global_id(0);
  const int out_blk_idx = get_global_id(1);
  const int input_chan_blk = (input_channel + 3) >> 2;

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

#if defined(USE_RELU) || defined(USE_RELUX) || defined(USE_PRELU) || defined(USE_TANH) || defined(USE_SIGMOID)
  result = do_activation(result, relux_max_limit, prelu_alpha);
#endif
  WRITE_IMAGET(output, (int2)(out_blk_idx, batch_idx), result);
}
