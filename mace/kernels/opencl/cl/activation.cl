#include <common.h>

__kernel void activation(__read_only image2d_t input,
                         __private const float relux_max_limit,
                         __private const float prelu_alpha,
                         __write_only image2d_t output) {
  const int ch_blk = get_global_id(0);
  const int w = get_global_id(1);
  const int hb = get_global_id(2);
  const int width = get_global_size(1);

  const int pos = mad24(ch_blk, width, w);
  DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(pos, hb));
  DATA_TYPE4 out = do_activation(in, relux_max_limit, prelu_alpha);
  WRITE_IMAGET(output, (int2)(pos, hb), out);
}

