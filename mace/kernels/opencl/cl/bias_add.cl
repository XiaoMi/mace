#include <common.h>
// Supported data types: half/float
__kernel void bias_add(
#ifndef USE_QUALCOMM_OPENCL_2_0
                       __private const int global_size_dim0,
                       __private const int global_size_dim1,
                       __private const int global_size_dim2,
#endif
                       __read_only image2d_t input,
                       __read_only image2d_t bias,
                       __write_only image2d_t output) {
  const int ch_blk = get_global_id(0);
  const int w = get_global_id(1);
  const int hb = get_global_id(2);

#ifndef USE_QUALCOMM_OPENCL_2_0
  if (ch_blk >= global_size_dim0 || w >= global_size_dim1
      || hb >= global_size_dim2) {
    return;
  }
  const int width = global_size_dim1;
#else
  const int width = get_global_size(1);
#endif

  const int pos = mad24(ch_blk, width, w);
  DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(pos, hb));
  DATA_TYPE4 bias_value = READ_IMAGET(bias, SAMPLER, (int2)(ch_blk, 0));
  DATA_TYPE4 out = in + bias_value;
  WRITE_IMAGET(output, (int2)(pos, hb), out);
}
