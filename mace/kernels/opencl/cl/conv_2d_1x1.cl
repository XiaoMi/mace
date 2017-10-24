/*
 * Split work item along output channels and pixels
 */
void kernel conv_2d_1x1_nchw(global const float *input, /* n, c, h, w */
                             global const float *filter, /* o, i, kh, kw */
                             global float *output, /* n, c, h, w */
                             private const int in_offset,
                             private const int out_offset,
                             private const int pixel_num,
                             private const int in_chan_num,
                             private const int out_chan_num) {
  int out_chan_blk = get_global_id(0);
  int out_pixel_blk = get_global_id(1);

  const int out_chan_begin = out_chan_blk * 4;
  const int out_chan_end = min(out_chan_begin + 4, out_chan_num);
  const int out_pixel_begin = out_pixel_blk * 4;
  const int out_pixel_end = min(out_pixel_begin + 4, pixel_num);

  const float *input_base = input + in_offset + out_pixel_begin;
  float *output_base = output + out_offset + out_pixel_begin;

  int pixels = out_pixel_end - out_pixel_begin;
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
