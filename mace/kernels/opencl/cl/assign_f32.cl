void kernel assign_f32(global float *vec, private const float value) {
  int idx = get_global_id(0);
  vec[idx] = value;
}

void kernel assign_vec_f32(global float *vec,
                           global float *values,
                           private int pixels) {
  int batch = get_global_id(0);
  int channel = get_global_id(1);
  int channels = get_global_size(1);
  float value = values[channel];
  float *ptr = vec + (batch * channels + channel) * pixels;
  for (int i = 0; i < pixels; ++i) {
    ptr[i] = value;
  }
}
