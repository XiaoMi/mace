#include <common.h>
// Supported data types: half/float
void kernel batch_norm(global const DATA_TYPE *input,
                       global const DATA_TYPE *scale,
                       global const DATA_TYPE *offset,
                       global const DATA_TYPE *mean,
                       global const DATA_TYPE *var,
                       global const DATA_TYPE *epsilon,
                       private const int pixels,
                       global DATA_TYPE *output,
                       __local VEC_DATA_TYPE(DATA_TYPE, 4) *new_scale,
                       __local VEC_DATA_TYPE(DATA_TYPE, 4) *new_offset) {
  const int batch = get_global_id(0);
  const int channel = get_global_id(1);
  const int channels = get_global_size(1);
  const int pixel_offset = get_global_id(2);
  const int local_channel = get_local_id(1);
  const int local_pixel_idx = get_local_id(2);

  if(local_pixel_idx == 0) {
    new_scale[local_channel] = (float4)(scale[channel] * rsqrt(var[channel] + *epsilon));
    new_offset[local_channel] = (float4)(offset[channel] - mean[channel] * new_scale[local_channel].x);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  const int image_offset = (batch * channels + channel) * pixels + pixel_offset*4;
  const DATA_TYPE *input_ptr = input + image_offset;
  DATA_TYPE *output_ptr = output + image_offset;
  const int end = (batch * channels + channel + 1) * pixels;
  if ((image_offset+4) > end) {
    for (int i = image_offset; i < end; ++i) {
      *output_ptr = new_scale[local_channel].x * *input_ptr + new_offset[local_channel].x;
      ++input_ptr;
      ++output_ptr;
    }
  } else {
    VEC_DATA_TYPE(DATA_TYPE, 4) values = vload4(0, input_ptr);
    values = values * new_scale[local_channel] + new_offset[local_channel];
    vstore4(values, 0, output_ptr);
  }
}

