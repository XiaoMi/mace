#include <common.h>

__kernel void addn(OUT_OF_RANGE_PARAMS
                   GLOBAL_WORK_GROUP_SIZE_DIM2
                   __read_only image2d_t input0, /* [c%4 * w * c/4, h * b] */
                   __read_only image2d_t input1,
#if INPUT_NUM > 2
                   __read_only image2d_t input2,
#endif
#if INPUT_NUM > 3
                   __read_only image2d_t input3,
#endif
                   __write_only image2d_t output) {
  const int w = get_global_id(0);
  const int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || hb >= global_size_dim1) return;
#endif

  DATA_TYPE4 in0 = READ_IMAGET(input0, SAMPLER, (int2)(w, hb));
  DATA_TYPE4 in1 = READ_IMAGET(input1, SAMPLER, (int2)(w, hb));
  DATA_TYPE4 out = in0 + in1;

#if INPUT_NUM > 2
  DATA_TYPE4 in2 = READ_IMAGET(input2, SAMPLER, (int2)(w, hb));
  out = out + in2;
#endif

#if INPUT_NUM > 3
  DATA_TYPE4 in3 = READ_IMAGET(input3, SAMPLER, (int2)(w, hb));
  out = out + in3;
#endif

  WRITE_IMAGET(output, (int2)(w, hb), out);
}

