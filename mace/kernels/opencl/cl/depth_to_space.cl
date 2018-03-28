#include <common.h>

__kernel void depth_to_space(__read_only image2d_t input,
                             __private const int block_size,
                             __private const int output_depth,
                             __write_only image2d_t output) {
  const int out_d = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_h = get_global_id(2);
  const int output_width = get_global_size(1);

  const int out_pos = mad24(out_d, output_width, out_w);
  const int input_width = output_width / block_size;

  const int in_h = out_h / block_size;
  const int offset_h = out_h % block_size;
  const int in_w = out_w / block_size;
  const int offset_w = out_w % block_size;
  const int offset_d = (offset_h * block_size + offset_w) * output_depth;
  const int in_d = out_d + offset_d;

  const int in_pos = mad24(in_d, input_width, in_w);
  DATA_TYPE4 in_data = READ_IMAGET(input, SAMPLER, (int2)(in_pos, in_h));
  WRITE_IMAGET(output, (int2)(out_pos, out_h), in_data);
}

__kernel void space_to_depth(__read_only image2d_t input,
                             __private const int block_size,
                             __private const int input_depth,
                             __write_only image2d_t output) {
  const int d = get_global_id(0);
  const int w = get_global_id(1);
  const int h = get_global_id(2);
  const int input_width = get_global_size(1);
  const int in_pos = mad24(d, input_width, w);
  const int output_width = input_width / block_size;

  const int out_h = h / block_size;
  const int offset_h = h % block_size;
  const int out_w = w / block_size;
  const int offset_w = w % block_size;
  const int offset_d = (offset_h * block_size + offset_w) * input_depth;
  const int out_d = d + offset_d;
  const int out_pos = mad24(out_d, output_width, out_w);

  DATA_TYPE4 in_data = READ_IMAGET(input, SAMPLER, (int2)(in_pos, h));

  WRITE_IMAGET(output, (int2)(out_pos, out_h), in_data);
}
