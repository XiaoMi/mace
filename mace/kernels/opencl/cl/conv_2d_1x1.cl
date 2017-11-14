#include <common.h>

__kernel void conv_2d_1x1_naive(__global const float *input, /* n, c, h, w */
                                __global const float *filter, /* o, i, kh, kw */
                                __global const float *bias, /* o */
                                __global float *output, /* n, c, h, w */
                                __private const int in_chan_num) {
  const int batch = get_global_id(0);
  const int channel = get_global_id(1);
  const int channels = get_global_size(1);
  const int pixel = get_global_id(2);
  const int pixels = get_global_size(2);

  float *output_ptr = output + (batch * channels + channel) * pixels;
  output_ptr[pixel] = bias[channel];

  for (int inc = 0; inc < in_chan_num; ++inc) {
    const float *input_ptr = input + (batch * in_chan_num + inc) * pixels + pixel;
    const float weights = filter[channel * in_chan_num + inc];
    float in = input_ptr[0];
    float out = output_ptr[0];
    out += in * weights;
    output_ptr[0] = out;
  }
}

__kernel void conv_2d_1x1_v2(__global const float *input, /* n, c, h, w */
                             __global const float *filter, /* o, i, kh, kw */
                             __global const float *bias, /* o */
                             __global float *output, /* n, c, h, w */
                             __private const int in_chan_num,
                             __private const int out_chan_num,
                             __private const int pixel_num) {
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

  int out_chan_len = out_chan_end - out_chan_begin;
  int pixel_len = out_pixel_end - out_pixel_begin;

  for (int out_chan = out_chan_begin; out_chan < out_chan_end; ++out_chan) {
    float *output_ptr = output_base + out_chan * pixel_num;
    float bias_value = bias == NULL ? 0 : bias[out_chan];
    for (int p = 0; p < pixel_len; ++p) {
      output_ptr[p] = bias_value;
    }
  }

  int in_chan = 0;
  if (pixel_len == 4) {
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
        #pragma unroll
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

      for (int p = 0; p < pixel_len; ++p) {
        float in = input_ptr[p];
        output_ptr[p] += in * weights;
      }
    }
  }
}

/* FIXME this is incomplete */
__kernel void conv_2d_1x1_v3(__read_only image3d_t input, /* n, c/4, h, w, 4 */
                             __global const float *filter, /* o, i, kh, kw */
                             __global const float *bias, /* o */
                             __write_only image3d_t output, /* n, c/4, h, w, 4 */
                             __private const int batch_num,
                             __private const int in_chan_num,
                             __private const int out_chan_num,
                             __private const int height,
                             __private const int width) {
  int out_chan_blk = get_global_id(0);
  int h = get_global_id(1);
  int w = get_global_id(2);


  int in_chan_blk_num = (in_chan_num + 3) / 4;
  int out_chan_blk_num = (out_chan_num + 3) / 4;

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  for (int batch = 0; batch < batch_num; ++batch) {
    float4 bias_value = vload4(out_chan_blk, bias);
    __private float4 out = bias_value;

    for (int in_chan_blk = 0; in_chan_blk < in_chan_blk_num; ++in_chan_blk) {
      int in_d = batch * in_chan_blk_num + in_chan_blk;
      float4 in = read_imagef(input, sampler, (int4)(in_d, h, w, 0));

      const float *filter_base = filter + (out_chan_blk << 2) * in_chan_num;
      float4 weights = vload4(in_chan_blk, filter_base);
      out.x += dot(in, weights);
      weights = vload4(in_chan_blk, filter_base + in_chan_num);
      out.y += dot(in, weights);
      weights = vload4(in_chan_blk, filter_base + in_chan_num * 2);
      out.z += dot(in, weights);
      weights = vload4(in_chan_blk, filter_base + in_chan_num * 3);
      out.w += dot(in, weights);
    }

    int out_d = batch * out_chan_blk_num + out_chan_blk;
    int4 out_coord = (int4)(out_d, h, w, 0);
    write_imagef(output, out_coord, out);
  }
}
