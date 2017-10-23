void kernel assign_v16_f32(global float *output,
                           private const float value,
                           private const int pixels) {
  int pixel_block = get_global_id(0);
  int pixel_offset = pixel_block * 16;

  float *output_ptr = output + pixel_offset;
  int remains = pixels - pixel_offset;
  if (remains >= 16) {
    for (int i = 0; i < 4; ++i) {
      vstore4(value, i, output_ptr);
    }
  } else {
    for (int i = 0; i < remains; ++i) {
      output_ptr[i] = value;
    }
  }
}

void kernel assign_3d_v16_f32(global float *output,
                              global const float *values,
                              private const int pixels) {
  int batch = get_global_id(0);
  int channel = get_global_id(1);
  int channels = get_global_size(1);
  int pixel_block = get_global_id(2);
  int pixel_offset = pixel_block * 16;

  float value = values[channel];
  float *output_ptr = output + (batch * channels + channel) * pixels +
                      pixel_offset;
  int remains = pixels - pixel_offset;
  if (remains >= 16) {
    for (int i = 0; i < 4; ++i) {
      vstore4(value, i, output_ptr);
    }
  } else {
    for (int i = 0; i < remains; ++i) {
      output_ptr[i] = value;
    }
  }
}
