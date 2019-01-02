#include <common.h>

__kernel void conv2d(BUFFER_OUT_OF_RANGE_PARAMS
                     GLOBAL_WORK_GROUP_SIZE_DIM2
                     __global IN_DATA_TYPE *padded_input,
                     __global IN_DATA_TYPE *filter,
#ifdef BIAS
                     __global IN_DATA_TYPE *bias,
#endif
                     __private const int in_height,
                     __private const int in_width,
                     __private const int in_chan,
                     __private const int filter_height,
                     __private const int filter_width,
                     __private const int filter_in_chan,
                     __private const int filter_chan_size,
                     __private const int out_height,
                     __private const int out_width,
                     __private const int out_chan,
                     __private const int stride_h,
                     __private const int stride_w,
                     __private const int dilated_h_offset,
                     __private const int dilated_w_offset,
                     __private const float relux_max_limit,
                     __private const float leakyrelu_coefficient,
                     __global OUT_DATA_TYPE *output) {
  const int out_wc_blk_idx = get_global_id(0);
  const int out_hb_idx = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (out_wc_blk_idx >= global_size_dim0 ||
      out_hb_idx >= global_size_dim1) {
    return;
  }
#endif
  const int out_chan_blk = (out_chan + 3) >> 2;

  const int out_width_blk_idx = out_wc_blk_idx / out_chan_blk;
  const int out_chan_blk_idx =
      out_wc_blk_idx - mul24(out_width_blk_idx, out_chan_blk);

  const int batch_idx = out_hb_idx / out_height;
  const int out_height_idx = out_hb_idx - mul24(batch_idx, out_height);
  const int out_width_idx = out_width_blk_idx << 2;
  const int out_chan_idx = out_chan_blk_idx << 2;

  const int in_height_idx = mul24(out_height_idx, stride_h);
  const int in_width_idx = mul24(out_width_idx, stride_w);
  const int strided_chan = mul24(in_chan, stride_w);

#ifdef BIAS
  DATA_TYPE4 out0 = CONVERT4(vload4(0, bias + out_chan_idx));
  DATA_TYPE4 out1 = out0;
  DATA_TYPE4 out2 = out0;
  DATA_TYPE4 out3 = out0;
#else
  DATA_TYPE4 out0 = 0;
  DATA_TYPE4 out1 = 0;
  DATA_TYPE4 out2 = 0;
  DATA_TYPE4 out3 = 0;
#endif

  const int in_offset_base = mul24(mad24(mad24(batch_idx, in_height, in_height_idx),
      in_width, in_width_idx), in_chan);
  int filter_offset_base = mul24(out_chan_blk_idx, filter_in_chan) << 2;
  DATA_TYPE4 in0, in1, in2, in3;
  DATA_TYPE4 w0, w1, w2, w3;

  for (int filter_h_idx = 0; filter_h_idx < filter_height; ++filter_h_idx) {
    int in_height_offset = mad24(filter_h_idx, dilated_h_offset, in_offset_base);
    for (int filter_w_idx = 0; filter_w_idx < filter_width; ++filter_w_idx) {
      int filter_offset = filter_offset_base;
      int in_offset = mad24(filter_w_idx, dilated_w_offset, in_height_offset);
      for (int in_chan_idx = 0; in_chan_idx < in_chan; in_chan_idx += 4) {
        w0 = CONVERT4(vload4(0, filter + filter_offset));
        w1 = CONVERT4(vload4(0, filter + filter_offset + 4));
        w2 = CONVERT4(vload4(0, filter + filter_offset + 8));
        w3 = CONVERT4(vload4(0, filter + filter_offset + 12));

        in0 = CONVERT4(vload4(0, padded_input + in_offset));
        in1 = CONVERT4(vload4(0, padded_input + in_offset + strided_chan));
        in2 = CONVERT4(vload4(0, padded_input + in_offset + (strided_chan << 1)));
        in3 = CONVERT4(vload4(0, padded_input + in_offset + strided_chan + (strided_chan << 1)));

        out0 = mad((DATA_TYPE4)(in0.x), w0, out0);
        out0 = mad((DATA_TYPE4)(in0.y), w1, out0);
        out0 = mad((DATA_TYPE4)(in0.z), w2, out0);
        out0 = mad((DATA_TYPE4)(in0.w), w3, out0);

        out1 = mad((DATA_TYPE4)(in1.x), w0, out1);
        out1 = mad((DATA_TYPE4)(in1.y), w1, out1);
        out1 = mad((DATA_TYPE4)(in1.z), w2, out1);
        out1 = mad((DATA_TYPE4)(in1.w), w3, out1);

        out2 = mad((DATA_TYPE4)(in2.x), w0, out2);
        out2 = mad((DATA_TYPE4)(in2.y), w1, out2);
        out2 = mad((DATA_TYPE4)(in2.z), w2, out2);
        out2 = mad((DATA_TYPE4)(in2.w), w3, out2);

        out3 = mad((DATA_TYPE4)(in3.x), w0, out3);
        out3 = mad((DATA_TYPE4)(in3.y), w1, out3);
        out3 = mad((DATA_TYPE4)(in3.z), w2, out3);
        out3 = mad((DATA_TYPE4)(in3.w), w3, out3);
        filter_offset += 16;
        in_offset += 4;
      }
      filter_offset_base += filter_chan_size;
    }
  }

#if  defined(USE_RELU) || defined(USE_LEAKYRELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
  out0 = do_activation(out0, relux_max_limit, leakyrelu_coefficient);
  out1 = do_activation(out1, relux_max_limit, leakyrelu_coefficient);
  out2 = do_activation(out2, relux_max_limit, leakyrelu_coefficient);
  out3 = do_activation(out3, relux_max_limit, leakyrelu_coefficient);
#endif

  int out_offset = mad24(mad24(mad24(batch_idx, out_height, out_height_idx),
      out_width, out_width_idx), out_chan, out_chan_idx);

#define WRITE_OUTPUT(i) \
  if (out_chan_idx + 4 > out_chan) {           \
    const int diff = out_chan - out_chan_idx;  \
    switch(diff) {                             \
      case 3:                                  \
        output[out_offset + 2] = CONVERT_TO(out##i.z, OUT_DATA_TYPE);     \
      case 2:                                  \
        output[out_offset + 1] = CONVERT_TO(out##i.y, OUT_DATA_TYPE);     \
      case 1:                                  \
        output[out_offset] = CONVERT_TO(out##i.x, OUT_DATA_TYPE);         \
    }                                          \
    CHECK_OUT_OF_RANGE_FOR_BUFFER(out_offset + diff - 1); \
  } else {                                     \
    VSTORE4(CONVERT_TO(out##i, OUT_DATA_TYPE4), output, out_offset);   \
  }

  WRITE_OUTPUT(0);
  if (out_width_idx + 1 >= out_width) return;
  out_offset += out_chan;
  WRITE_OUTPUT(1);
  if (out_width_idx + 2 >= out_width) return;
  out_offset += out_chan;
  WRITE_OUTPUT(2);
  if (out_width_idx + 3 >= out_width) return;
  out_offset += out_chan;
  WRITE_OUTPUT(3);
#undef WRITE_OUTPUT

}
