/*
 * Split work item along output channels and pixels
 */
void kernel conv_2d_1x1_naive(global const float *input, /* n, c, h, w */
                              global const float *filter, /* o, i, kh, kw */
                              global float *output, /* n, c, h, w */
                              private const int in_offset,
                              private const int out_offset,
                              private const int pixel_num,
                              private const int in_chan_num,
                              private const int out_chan_num) {
  int out_chan_blk = get_global_id(0);
  int out_pixel_blk = get_global_id(1);

  const int out_chan_begin = out_chan_blk << 2;
  const int out_chan_end = min(out_chan_begin + 4, out_chan_num);
  const int out_pixel_begin = out_pixel_blk << 3;
  const int out_pixel_end = min(out_pixel_begin + 8, pixel_num);

  const float *input_base = input + in_offset + out_pixel_begin;
  float *output_base = output + out_offset + out_pixel_begin;
  int pixels = out_pixel_end - out_pixel_begin;

  for (int in_chan = 0; in_chan < in_chan_num; ++in_chan) {
    const float *input_ptr = input_base + in_chan * pixel_num;
    if (pixels == 8) {
      /* TODO fix '#pragma unroll' build error */
      for (int out_chan = out_chan_begin; out_chan < out_chan_end; ++out_chan) {
        float weights = filter[out_chan * in_chan_num + in_chan];
        float *output_ptr = output_base + out_chan * pixel_num;
        for (int p = 0; p < 2; ++p) {
          float4 in = vload4(p, input_ptr);
          float4 out = vload4(p, output_ptr);
          out += in * weights;
          vstore4(out, p, output_ptr);
        }
      }
    } else {
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
}
