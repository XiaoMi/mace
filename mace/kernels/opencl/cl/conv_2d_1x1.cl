#include <common.h>

#define vec_conv_2d_1x1_s1                    \
  VEC_DATA_TYPE(DATA_TYPE,4) in0 = vload4(0, input_ptr);                   \
  VEC_DATA_TYPE(DATA_TYPE,4) in1 = vload4(0, input_ptr + in_pixel);        \
  VEC_DATA_TYPE(DATA_TYPE,4) in2 = vload4(0, input_ptr + 2 * in_pixel);    \
  VEC_DATA_TYPE(DATA_TYPE,4) in3 = vload4(0, input_ptr + 3 * in_pixel);


#define vec_conv_2d_1x1_s2                    \
  VEC_DATA_TYPE(DATA_TYPE,4) in00 = vload4(0, input_ptr);                   \
  VEC_DATA_TYPE(DATA_TYPE,3) in01 = vload3(0, input_ptr + 4);               \
  VEC_DATA_TYPE(DATA_TYPE,4) in10 = vload4(0, input_ptr + in_pixel);        \
  VEC_DATA_TYPE(DATA_TYPE,3) in11 = vload3(0, input_ptr + in_pixel + 4);    \
  VEC_DATA_TYPE(DATA_TYPE,4) in20 = vload4(0, input_ptr + 2 * in_pixel);    \
  VEC_DATA_TYPE(DATA_TYPE,3) in21 = vload3(0, input_ptr + 2 * in_pixel + 4);\
  VEC_DATA_TYPE(DATA_TYPE,4) in30 = vload4(0, input_ptr + 3 * in_pixel);    \
  VEC_DATA_TYPE(DATA_TYPE,3) in31 = vload3(0, input_ptr + 3 * in_pixel + 4); \
  VEC_DATA_TYPE(DATA_TYPE,4) in0 = (VEC_DATA_TYPE(DATA_TYPE,4))(in00.s02, in01.s02);            \
  VEC_DATA_TYPE(DATA_TYPE,4) in1 = (VEC_DATA_TYPE(DATA_TYPE,4))(in10.s02, in11.s02);            \
  VEC_DATA_TYPE(DATA_TYPE,4) in2 = (VEC_DATA_TYPE(DATA_TYPE,4))(in20.s02, in21.s02);            \
  VEC_DATA_TYPE(DATA_TYPE,4) in3 = (VEC_DATA_TYPE(DATA_TYPE,4))(in30.s02, in31.s02);


#define vec_conv_2d_1x1_compute_loop  \
  for (int oc = 0; oc < 4; ++oc) {                             \
    VEC_DATA_TYPE(DATA_TYPE,4) weights = vload4(0, filter_ptr + oc * in_chan_num); \
    VEC_DATA_TYPE(DATA_TYPE,4) out = vload4(0, output_ptr + oc * out_pixel);       \
    out += in0 * weights.x;                                    \
    out += in1 * weights.y;                                     \
    out += in2 * weights.z;                                     \
    out += in3 * weights.w;                                     \
    vstore4(out, 0, output_ptr + oc * out_pixel);               \
  }

#define vec_conv_2d_1x1_compute  \
    VEC_DATA_TYPE(DATA_TYPE,4) weights = vload4(0, filter_ptr); \
    VEC_DATA_TYPE(DATA_TYPE,4) out = vload4(0, output_ptr);       \
    out += in0 * weights.x;                                    \
    out += in1 * weights.y;                                     \
    out += in2 * weights.z;                                     \
    out += in3 * weights.w;                                     \
    vstore4(out, 0, output_ptr);

// Supported data type: half/float
__kernel void conv_2d_1x1_v2(__global const DATA_TYPE *input, /* n, c, h, w */
                             __global const DATA_TYPE *filter, /* o, i, kh, kw */
#ifdef BIAS
                             __global const DATA_TYPE *bias, /* o */
#endif /* defined(BIAS) */
                             __global DATA_TYPE *output, /* n, c, h, w */
                             __private const int in_chan_num,
                             __private const int out_chan_num,
                             __private const int in_height,
                             __private const int in_width,
                             __private const int out_height,
                             __private const int out_width) {
  int batch = get_global_id(0);
  int out_chan_blk = get_global_id(1);
  int out_pixel_blk = get_global_id(2);

  const int in_pixel = in_height * in_width;
  const int out_pixel = out_height * out_width;

  const int round_out_width = (out_width + 3) / 4;
  const int out_pixel_height = out_pixel_blk / round_out_width;
  const int out_pixel_width = out_pixel_blk % round_out_width;

  const int out_chan_begin = out_chan_blk * 4;
  const int out_chan_end = min(out_chan_begin + 4, out_chan_num);
  const int out_pixel_begin = out_pixel_height * out_width + out_pixel_width * 4;
  const int out_pixel_end = min(out_pixel_begin + 4, (out_pixel_height + 1) * out_width);

#ifdef STRIDE_1
  const int stride = 1;
#else
  const int stride = 2;
#endif
  const int in_pixel_begin = out_pixel_height * stride * in_width + out_pixel_width * stride * 4;

  const int in_offset = batch * in_chan_num * in_pixel;
  const int out_offset = batch * out_chan_num * out_pixel;

  const DATA_TYPE *input_base = input + in_offset + in_pixel_begin;
  DATA_TYPE *output_base = output + out_offset + out_pixel_begin;

  int out_chan_len = out_chan_end - out_chan_begin;
  int pixel_len = out_pixel_end - out_pixel_begin;

  for (int out_chan = out_chan_begin; out_chan < out_chan_end; ++out_chan) {
    DATA_TYPE *output_ptr = output_base + out_chan * out_pixel;
#ifdef BIAS
    DATA_TYPE bias_value = bias[out_chan];
#else
    DATA_TYPE bias_value = 0;
#endif
    for (int p = 0; p < pixel_len; ++p) {
      output_ptr[p] = bias_value;
    }
  }

  int in_chan = 0;
  if (pixel_len == 4) {
    for (; in_chan + 3 < in_chan_num; in_chan += 4) {
      const DATA_TYPE *input_ptr = input_base + in_chan * in_pixel;
      int out_chan = out_chan_begin;
      for (; out_chan + 3 < out_chan_end; out_chan += 4) {
        const DATA_TYPE* filter_ptr = filter + out_chan * in_chan_num + in_chan;
        DATA_TYPE *output_ptr = output_base + out_chan * out_pixel;
#ifdef STRIDE_1
        vec_conv_2d_1x1_s1;
#else
        vec_conv_2d_1x1_s2;
#endif
        vec_conv_2d_1x1_compute_loop;
      }
      for (; out_chan < out_chan_end; ++out_chan) {
        const DATA_TYPE* filter_ptr = filter + out_chan * in_chan_num + in_chan;
        DATA_TYPE *output_ptr = output_base + out_chan * out_pixel;
#ifdef STRIDE_1
        vec_conv_2d_1x1_s1;
#else
        vec_conv_2d_1x1_s2;
#endif
        vec_conv_2d_1x1_compute;
      }
    }
  }

  for (; in_chan < in_chan_num; ++in_chan) {
    const DATA_TYPE *input_ptr = input_base + in_chan * in_pixel;
    for (int out_chan = out_chan_begin; out_chan < out_chan_end; ++out_chan) {
      DATA_TYPE weights = filter[out_chan * in_chan_num + in_chan];
      DATA_TYPE *output_ptr = output_base + out_chan * out_pixel;

      for (int p = 0; p < pixel_len; ++p) {
        float in = input_ptr[p*stride];
        output_ptr[p] += in * weights;
      }
    }
  }
}

// TODO : validation
__kernel void conv_2d_1x1(__read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
                          __read_only image2d_t filter, /* cout%4 * cin, cout/4 */
                          __read_only image2d_t bias, /* cout%4 * cout/4 */
                          __write_only image2d_t output,
                          __private const int in_ch_blks,
                          __private const int width) {
  const int out_ch_blk = get_global_id(0);
  const int out_w_blk = get_global_id(1);
  const int out_w_blks = get_global_size(1);
  const int out_hb = get_global_id(2);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  half4 bias_value = read_imageh(bias, sampler, (int2)(out_w_blk, 1));
  half4 out0 = (half4)(bias_value.x, bias_value.x, bias_value.x, bias_value.x);
  half4 out1 = (half4)(bias_value.y, bias_value.y, bias_value.y, bias_value.y);
  half4 out2 = (half4)(bias_value.z, bias_value.z, bias_value.z, bias_value.z);
  half4 out3 = (half4)(bias_value.w, bias_value.w, bias_value.w, bias_value.w);

  // Unrolling this loop hurt perfmance
  int in_x_base = 0;
  for (int in_ch_blk = 0; in_ch_blk < in_ch_blks; ++in_ch_blk) {
    const int in_x0 = in_x_base + out_w_blk;;
    const int in_x1 = in_x0 + out_w_blks;
    const int in_x2 = in_x1 + out_w_blks;
    const int in_x3 = in_x2 + out_w_blks;
    in_x_base += width;

    const half4 in0 = read_imageh(input, sampler, (int2)(in_x0, out_hb));
    const half4 in1 = read_imageh(input, sampler, (int2)(in_x1, out_hb));
    const half4 in2 = read_imageh(input, sampler, (int2)(in_x2, out_hb));
    const half4 in3 = read_imageh(input, sampler, (int2)(in_x3, out_hb));

    // The order matters, load input first then load filter, why?
    const int filter_x0 = in_ch_blk << 2;
    const half4 weights0 = read_imageh(filter, sampler, (int2)(filter_x0, out_ch_blk));
    const half4 weights1 = read_imageh(filter, sampler, (int2)(filter_x0 + 1, out_ch_blk));
    const half4 weights2 = read_imageh(filter, sampler, (int2)(filter_x0 + 2, out_ch_blk));
    const half4 weights3 = read_imageh(filter, sampler, (int2)(filter_x0 + 3, out_ch_blk));
    // Will prefetch L2 improve performance? How to pretch image data?

    // Interleaving load and mul does not improve performance as expected
    out0 += in0.x * weights0;
    out1 += in1.x * weights0;
    out2 += in2.x * weights0;
    out3 += in3.x * weights0;

    out0 += in0.y * weights1;
    out1 += in1.y * weights1;
    out2 += in2.y * weights1;
    out3 += in3.y * weights1;

    out0 += in0.z * weights2;
    out1 += in1.z * weights2;
    out2 += in2.z * weights2;
    out3 += in3.z * weights2;

    out0 += in0.w * weights3;
    out1 += in1.w * weights3;
    out2 += in2.w * weights3;
    out3 += in3.w * weights3;
  }

  const int out_x_offset = out_ch_blk * width;
  const int w0 = out_w_blk;
  write_imageh(output, (int2)(out_x_offset + w0, out_hb), out0);

  const int w1 = w0 + out_w_blks;
  if (w1 >= width) return;
  write_imageh(output, (int2)(out_x_offset + w1, out_hb), out1);

  const int w2 = w1 + out_w_blks;
  if (w2 >= width) return;
  write_imageh(output, (int2)(out_x_offset + w2, out_hb), out2);

  const int w3 = w2 + out_w_blks;
  if (w3 >= width) return;
  write_imageh(output, (int2)(out_x_offset + w3, out_hb), out3);
}


