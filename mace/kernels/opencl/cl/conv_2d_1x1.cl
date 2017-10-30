void kernel conv_2d_1x1_naive(global const float *input, /* n, c, h, w */
                              global const float *filter, /* o, i, kh, kw */
                              global const float *bias, /* o */
                              global float *output, /* n, c, h, w */
                              private const int input_channels) {
  const int batch = get_global_id(0);
  const int channel = get_global_id(1);
  const int channels = get_global_size(1);
  const int pixel = get_global_id(2);
  const int pixels = get_global_size(2);


  float *output_ptr = output + (batch * channels + channel) * pixels;
  output_ptr[pixel] = bias[channel];

  for (int inc = 0; inc < input_channels; ++inc) {
    const float *input_ptr = input + (batch * input_channels + inc) * pixels + pixel;
    const float weights = filter[channel * input_channels + inc];
    float in = input_ptr[0];
    float out = output_ptr[0];
    out += in * weights;
    output_ptr[0] = out;
  }
}

void kernel conv_2d_1x1_v2(global const float *input, /* n, c, h, w */
                           global const float *filter, /* o, i, kh, kw */
                           global const float *bias, /* o */
                           global float *output, /* n, c, h, w */
                           private const int in_chan_num,
                           private const int out_chan_num,
                           private const int pixel_num) {
  int batch = get_global_id(0);
  int out_chan_blk = get_global_id(1);
  int out_pixel_blk = get_global_id(2);

  const int out_chan_begin = out_chan_blk * 4;
  const int out_chan_end = min(out_chan_begin + 4, out_chan_num);
  const int out_pixel_begin = out_pixel_blk * 4;
  const int out_pixel_end = min(out_pixel_begin + 4, pixel_num);

  const int in_offset = batch * in_chan_num * pixel_num;
  const int out_offset = batch * out_chan_num * pixel_num;
  const float *input_base = input + in_offset + out_pixel_begin;
  float *output_base = output + out_offset + out_pixel_begin;

  int pixels = out_pixel_end - out_pixel_begin;

  for (int out_chan = out_chan_begin; out_chan < out_chan_end; ++out_chan) {
    float bias_value = bias[out_chan];
    float *output_ptr = output_base + out_chan * pixel_num;
    for (int p = 0; p < pixels; ++p) {
      output_ptr[p] = bias_value;
    }
  }

  int in_chan = 0;
  if (pixels == 4) {
    for (; in_chan + 3 < in_chan_num; in_chan += 4) {
      const float *input_ptr = input_base + in_chan * pixel_num;
      int out_chan = out_chan_begin;
      for (; out_chan + 3 < out_chan_end; out_chan += 4) {
        const float* filter_ptr = filter + out_chan * in_chan_num + in_chan;
        float *output_ptr = output_base + out_chan * pixel_num;
        float4 in0 = vload4(0, input_ptr);
        float4 in1 = vload4(0, input_ptr + pixel_num);
        float4 in2 = vload4(0, input_ptr + 2 * pixel_num);
        float4 in3 = vload4(0, input_ptr + 3 * pixel_num);
        for (int oc = 0; oc < 4; ++oc) {
          float4 weights = vload4(0, filter_ptr + oc * in_chan_num);
          float4 out = vload4(0, output_ptr + oc * pixel_num);
          out += in0 * weights.x;
          out += in1 * weights.y;
          out += in2 * weights.z;
          out += in3 * weights.w;
          vstore4(out, 0, output_ptr + oc * pixel_num);
        }
      }
      for (; out_chan < out_chan_end; ++out_chan) {
        const float* filter_ptr = filter + out_chan * in_chan_num + in_chan;
        float *output_ptr = output_base + out_chan * pixel_num;
        float4 weights = vload4(0, filter_ptr);
        float4 in0 = vload4(0, input_ptr);
        float4 in1 = vload4(0, input_ptr + pixel_num);
        float4 in2 = vload4(0, input_ptr + 2 * pixel_num);
        float4 in3 = vload4(0, input_ptr + 3 * pixel_num);
        float4 out = vload4(0, output_ptr);
        out += in0 * weights.x;
        out += in1 * weights.y;
        out += in2 * weights.z;
        out += in3 * weights.w;
        vstore4(out, 0, output_ptr);
      }
    }
  }

  for (; in_chan < in_chan_num; ++in_chan) {
    const float *input_ptr = input_base + in_chan * pixel_num;
    for (int out_chan = out_chan_begin; out_chan < out_chan_end; ++out_chan) {
      float weights = filter[out_chan * in_chan_num + in_chan];
      float *output_ptr = output_base + out_chan * pixel_num;

      for (int p = 0; p < pixels; ++p) {
        float in = input_ptr[p];
        float out = output_ptr[p];
        out += in * weights;
        output_ptr[p] = out;
      }
    }
  }
}
