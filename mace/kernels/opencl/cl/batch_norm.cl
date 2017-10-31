void kernel batch_norm(global const float *input,
                       global const float *scale,
                       global const float *offset,
                       global const float *mean,
                       global const float *var,
                       global const float *epsilon,
                       private const uint pixels,
                       global float *output,
                       __local float4 *new_scale,
                       __local float4 *new_offset) {
  const int batch = get_global_id(0);
  const int channel = get_global_id(1);
  const int channels = get_global_size(1);
  const int pixel_offset = get_global_id(2);
  const unsigned int local_channel = get_local_id(1);
  const int local_pixel_idx = get_local_id(2);

  if(local_pixel_idx == 0) {
    new_scale[local_channel] = (float4)(scale[channel] * rsqrt(var[channel] + *epsilon));
    new_offset[local_channel] = (float4)(offset[channel] - mean[channel] * new_scale[local_channel].x);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  const int image_offset = (batch * channels + channel) * pixels + pixel_offset*4;
  const float *input_ptr = input + image_offset;
  float *output_ptr = output + image_offset;
  const int end = (batch * channels + channel + 1) * pixels;
  if ((image_offset+4) > end) {
    for (int i = image_offset; i < end; ++i) {
      *output_ptr = new_scale[local_channel].x * *input_ptr + new_offset[local_channel].x;
      ++input_ptr;
      ++output_ptr;
    }
  } else {
    float4 values = vload4(0, input_ptr);
    values = values * new_scale[local_channel] + new_offset[local_channel];
    vstore4(values, 0, output_ptr);
  }
}

