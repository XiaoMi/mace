#include <common.h>

__kernel void argmax(OUT_OF_RANGE_PARAMS
                     GLOBAL_WORK_GROUP_SIZE_DIM3
                     __read_only image2d_t input,
                     __private const int channel_blocks,
                     __write_only image2d_t output) {

  const int w = get_global_id(1);
  const int h = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim1 || h >= global_size_dim2) {
    return;
  }
#endif
  const int width = global_size_dim1;

  float idx = 0;
  float max = -1;

  for (int i = 0; i < channel_blocks; ++i) {
    const int pos = mad24(i, width, w);
    DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(pos, h));

    if (in.x > max) {
      max = in.x;
      idx = (i * 4);
    }
    if (in.y > max) {
      max = in.y;
      idx = (i * 4) + 1;
    }
    if (in.z > max) {
      max = in.z;
      idx = (i * 4) + 2;
    }
    if (in.w > max) {
      max = in.w;
      idx = (i * 4) + 3;
    }
  }

  const int pos = mad24(0, width, w);
  WRITE_IMAGET(output, (int2)(pos, h), (float4)(idx, 0, 0, 0));
}
