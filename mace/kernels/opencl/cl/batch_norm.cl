void kernel batch_norm(global const float *input,
                       global const float *scale,
                       global const float *offset,
                       global const float *mean,
                       global const float *var,
                       global const float *epsilon,
                       private const uint pixels,
                       global float *output,
                       __local float *new_scale,
                       __local float *new_offset) {
  const int batch = get_global_id(0);
  const int channel = get_global_id(1);
  const int channels = get_global_size(1);
  const int pixel_offset = get_global_id(2);
  const unsigned int local_channel = get_local_id(1);
  const int local_pixel_idx = get_local_id(2);

  if(local_pixel_idx == 0) {
    new_scale[local_channel] = scale[channel] * rsqrt(var[channel] + *epsilon);
    new_offset[local_channel] = offset[channel] - mean[channel] * new_scale[local_channel];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  const int sample_offset = (batch * channels + channel) * pixels + pixel_offset;
  const float *input_ptr = input + sample_offset;
  float *output_ptr = output + sample_offset;
  *output_ptr = new_scale[local_channel] * *input_ptr + new_offset[local_channel];
}

