void kernel batch_norm(global const float *input,
                       global const float *scale,
                       global const float *offset,
                       global const float *mean,
                       global const float *var,
                       global const float *epsilon,
                       private const int channels,
                       private const int pixels,
                       global float *output) {
  int idx = get_global_id(0);
  int channel = (idx % (channels * pixels)) / pixels;

  const float *input_ptr = input + idx;
  const float new_scale = scale[channel] * rsqrt(var[channel] + *epsilon);
  const float new_offset = offset[channel] - mean[channel] * new_scale;
  float *output_ptr = output + idx;
  *output_ptr = new_scale * *input_ptr + new_offset;
}

