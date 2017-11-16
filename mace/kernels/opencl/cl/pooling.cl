#include <common.h>

VEC_DATA_TYPE(DATA_TYPE,4) vec_pooling_3_s1(const DATA_TYPE *input_ptr, const int in_width) {
  VEC_DATA_TYPE(DATA_TYPE,4) row00 = vload4(0, input_ptr);
  VEC_DATA_TYPE(DATA_TYPE,2) row01 = vload2(0, input_ptr + 4);
  VEC_DATA_TYPE(DATA_TYPE,4) row10 = vload4(0, input_ptr + in_width);
  VEC_DATA_TYPE(DATA_TYPE,2) row11 = vload2(0, input_ptr + in_width + 4);
  VEC_DATA_TYPE(DATA_TYPE,4) row20 = vload4(0, input_ptr + in_width * 2);
  VEC_DATA_TYPE(DATA_TYPE,2) row21 = vload2(0, input_ptr + in_width * 2 + 4);

  VEC_DATA_TYPE(DATA_TYPE,8) data00 = (VEC_DATA_TYPE(DATA_TYPE,8))(row00.s01212323);
  VEC_DATA_TYPE(DATA_TYPE,4) data01 = (VEC_DATA_TYPE(DATA_TYPE,4))(row01.s0, row00.s3, row01.s01);
  VEC_DATA_TYPE(DATA_TYPE,8) data10 = (VEC_DATA_TYPE(DATA_TYPE,8))(row10.s01212323);
  VEC_DATA_TYPE(DATA_TYPE,4) data11 = (VEC_DATA_TYPE(DATA_TYPE,4))(row11.s0, row10.s3, row11.s01);
  VEC_DATA_TYPE(DATA_TYPE,8) data20 = (VEC_DATA_TYPE(DATA_TYPE,8))(row20.s01212323);
  VEC_DATA_TYPE(DATA_TYPE,4) data21 = (VEC_DATA_TYPE(DATA_TYPE,4))(row21.s0, row20.s3, row21.s01);

  VEC_DATA_TYPE(DATA_TYPE,8) left = fmax(fmax(data00, data10), data20);
  VEC_DATA_TYPE(DATA_TYPE,4) right = fmax(fmax(data01, data11), data21);

  VEC_DATA_TYPE(DATA_TYPE,4) res = fmax((VEC_DATA_TYPE(DATA_TYPE,4))(left.s036, right.s1),
                                        (VEC_DATA_TYPE(DATA_TYPE,4))(left.s147, right.s2));
  res = fmax(res, (VEC_DATA_TYPE(DATA_TYPE,4))(left.s25, right.s03));

  return res;
}

VEC_DATA_TYPE(DATA_TYPE,4) vec_pooling_3_s2(const DATA_TYPE *input_ptr, const int in_width) {
  VEC_DATA_TYPE(DATA_TYPE,8) row00 = vload8(0, input_ptr);
  DATA_TYPE row01 = *(input_ptr + 8);
  VEC_DATA_TYPE(DATA_TYPE,8) row10 = vload8(0, input_ptr + in_width);
  DATA_TYPE row11 = *(input_ptr + in_width + 8);
  VEC_DATA_TYPE(DATA_TYPE,8) row20 = vload8(0, input_ptr + in_width * 2);
  DATA_TYPE row21 = *(input_ptr + in_width * 2 + 8);

  VEC_DATA_TYPE(DATA_TYPE,8) data00 = (VEC_DATA_TYPE(DATA_TYPE,8))(row00.s01223445);
  VEC_DATA_TYPE(DATA_TYPE,4) data01 = (VEC_DATA_TYPE(DATA_TYPE,4))(row00.s667, row01);
  VEC_DATA_TYPE(DATA_TYPE,8) data10 = (VEC_DATA_TYPE(DATA_TYPE,8))(row10.s01223445);
  VEC_DATA_TYPE(DATA_TYPE,4) data11 = (VEC_DATA_TYPE(DATA_TYPE,4))(row10.s667, row11);
  VEC_DATA_TYPE(DATA_TYPE,8) data20 = (VEC_DATA_TYPE(DATA_TYPE,8))(row20.s01223445);
  VEC_DATA_TYPE(DATA_TYPE,4) data21 = (VEC_DATA_TYPE(DATA_TYPE,4))(row20.s667, row21);

  VEC_DATA_TYPE(DATA_TYPE,8) left = fmax(fmax(data00, data10), data20);
  VEC_DATA_TYPE(DATA_TYPE,4) right = fmax(fmax(data01, data11), data21);

  VEC_DATA_TYPE(DATA_TYPE,4) res = fmax((VEC_DATA_TYPE(DATA_TYPE,4))(left.s036, right.s1),
                                        (VEC_DATA_TYPE(DATA_TYPE,4))(left.s147, right.s2));
  res = fmax(res, (VEC_DATA_TYPE(DATA_TYPE,4))(left.s25, right.s03));

  return res;
}

DATA_TYPE inner_pooling_3(const DATA_TYPE *input_ptr, const int in_width) {
  VEC_DATA_TYPE(DATA_TYPE,3) row0 = vload3(0, input_ptr);
  VEC_DATA_TYPE(DATA_TYPE,3) row1 = vload3(0, input_ptr + in_width);
  VEC_DATA_TYPE(DATA_TYPE,3) row2 = vload3(0, input_ptr + in_width * 2);

  VEC_DATA_TYPE(DATA_TYPE,3) data = fmax(fmax(row0, row1), row2);

  DATA_TYPE res = fmax(fmax(data.s0, data.s1), data.s2);
  return res;
}

// Supported data type: half/float
__kernel void pooling3(__global const DATA_TYPE *input, /* n, c, h, w */
                       __private const int in_height,
                       __private const int in_width,
                       __private const int out_chan_num,
                       __private const int out_height,
                       __private const int out_width,
                       __private const int stride,
                       __global DATA_TYPE *output) {
  int batch = get_global_id(0);
  int out_chan_blk = get_global_id(1);
  int out_pixel_blk = get_global_id(2);

  const int round_out_width = (out_width + 3) / 4;
  const int out_pixel_height = out_pixel_blk / round_out_width;
  const int out_pixel_width = out_pixel_blk % round_out_width;

  const int out_chan_begin = out_chan_blk * 4;
  const int out_chan_end = min(out_chan_begin + 4, out_chan_num);
  const int out_pixel_begin = out_pixel_height * out_width + out_pixel_width * 4;
  const int out_pixel_end = min(out_pixel_begin + 4, (out_pixel_height + 1) * out_width);
  const int in_pixel_begin = out_pixel_height * stride * in_width + out_pixel_width * stride * 4;

  const int in_pixel = in_height * in_width;
  const int out_pixel = out_height * out_width;

  const int in_offset = batch * out_chan_num * in_pixel;
  const int out_offset = batch * out_chan_num * out_pixel;
  const DATA_TYPE *input_base = input + in_offset + in_pixel_begin;
  DATA_TYPE *output_base = output + out_offset + out_pixel_begin;

  const int pixels = out_pixel_end - out_pixel_begin;

  for (int i = out_chan_begin; i < out_chan_end; ++i) {
    const DATA_TYPE *input_ptr = input_base + i * in_pixel;
    DATA_TYPE *output_ptr = output_base + i * out_pixel;
    if (pixels == 4) {
      VEC_DATA_TYPE(DATA_TYPE,4) res;
#ifdef STRIDE_1
      res = vec_pooling_3_s1(input_ptr, in_width);
#else
      res = vec_pooling_3_s2(input_ptr, in_width);
#endif
      vstore4(res, 0, output_ptr);
    } else {
      for (int p = 0; p < pixels; ++p) {
        output_ptr[p] = inner_pooling_3(input_ptr, in_width);
        input_ptr += stride;
      }
    }
  }
}

int calculate_avg_block_size(const int pos_h,
                             const int pos_w,
                             const int pool_size,
                             const int pad_h,
                             const int pad_w,
                             const int h_size,
                             const int w_size) {
  const int h_start = max(0, pos_h - pad_h);
  const int w_start = max(0, pos_w - pad_w);
  const int h_end = min(pos_h + pool_size - pad_h, h_size);
  const int w_end = min(pos_w + pool_size - pad_w, w_size);
  return (h_end - h_start) * (w_end - w_start);
}

// Supported data type: half/float
__kernel void poolingn(__global const DATA_TYPE *input, /* n, c, h, w */
                       __private const int in_height,
                       __private const int in_width,
                       __private const int out_chan_num,
                       __private const int out_height,
                       __private const int out_width,
                       __private const int stride,
                       __private const int pad_h,
                       __private const int pad_w,
                       __private const int pooling_size,
                       __global DATA_TYPE *output) {
  int batch = get_global_id(0);
  int out_chan_idx = get_global_id(1);
  int out_pixel_idx = get_global_id(2);

  const int out_pixel_height = out_pixel_idx / out_width;
  const int out_pixel_width = out_pixel_idx % out_width;

  const int out_chan_begin = out_chan_idx * 4;
  const int out_chan_end = min(out_chan_begin + 4, out_chan_num);
  const int in_pixel_idx = out_pixel_height * stride * in_width
                             + out_pixel_width * stride;

  const int in_pixel = in_height * in_width;
  const int out_pixel = out_height * out_width;

  const int in_offset = batch * out_chan_num * in_pixel;
  const int out_offset = batch * out_chan_num * out_pixel;
  const DATA_TYPE *input_base = input + in_offset + in_pixel_idx;
  DATA_TYPE *output_base = output + out_offset + out_pixel_idx;

  const int block_size = calculate_avg_block_size(
                            out_pixel_height * stride,
                            out_pixel_width * stride,
                            pooling_size,
                            pad_h/2,
                            pad_w/2,
                            in_height - pad_h,
                            in_width - pad_w);
  for (int i = out_chan_begin; i < out_chan_end; ++i) {
    VEC_DATA_TYPE(DATA_TYPE,8) sum8 = 0.0f;
    DATA_TYPE sum1 = 0.0f;
    DATA_TYPE *output_ptr = output_base + i * out_pixel;
    for (int y = 0; y < pooling_size; ++y) {
      const DATA_TYPE *input_ptr = input_base + i * in_pixel + y * in_width;
      int x = 0;
      for (; x < (pooling_size-8); x += 8) {
        VEC_DATA_TYPE(DATA_TYPE,8) data = vload8(0, input_ptr);
        sum8 += data;
        input_ptr += 8;
      }
      for (; x < pooling_size; ++x) {
        sum1 += *input_ptr;
        input_ptr++;
      }
    }
    VEC_DATA_TYPE(DATA_TYPE,4) sum4 = sum8.s0123 + sum8.s4567;
    VEC_DATA_TYPE(DATA_TYPE,2) sum2 = sum4.s01 + sum4.s23;

    *output_ptr = (sum2.s0 + sum2.s1 + sum1) / block_size;
  }
}
