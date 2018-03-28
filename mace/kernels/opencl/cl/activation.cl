#include <common.h>

__kernel void activation(__read_only image2d_t input,
#ifdef USE_PRELU
                         __read_only image2d_t alpha,
#endif
                         __private const float relux_max_limit,
#ifndef USE_QUALCOMM_OPENCL_2_0
                         __write_only image2d_t output,
                         __private const int global_size_dim0,
                         __private const int global_size_dim1,
                         __private const int global_size_dim2) {
#else
                         __write_only image2d_t output) {
#endif

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
#ifdef USE_PRELU
  DATA_TYPE4 prelu_alpha = READ_IMAGET(alpha, SAMPLER, (int2)(ch_blk, 0));
  DATA_TYPE4 out = do_activation(in, prelu_alpha, relux_max_limit);
#else
  DATA_TYPE4 out = do_activation(in, relux_max_limit);
#endif
  WRITE_IMAGET(output, (int2)(pos, hb), out);
}

