#include <common.h>

__kernel void eltwise(
                      UNIFORM_WORK_GROUP_SIZE_PARAMS_IN_DIM_2
                      __read_only image2d_t input0, /* [c%4 * w * c/4, h * b] */
                      __read_only image2d_t input1,
#ifdef COEFF_SUM
                      __private const float coeff0,
                      __private const float coeff1,
#endif
                      __write_only image2d_t output) {
  const int w = get_global_id(0);
  const int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || hb >= global_size_dim1) return;
#endif

  DATA_TYPE4 in0 = READ_IMAGET(input0, SAMPLER, (int2)(w, hb));
  DATA_TYPE4 in1 = READ_IMAGET(input1, SAMPLER, (int2)(w, hb));
  DATA_TYPE4 out;
#if ELTWISE_TYPE == 0
  out = in0 * in1;
#elif ELTWISE_TYPE == 1

#ifdef COEFF_SUM
  out = mad(coeff0, in0, mad(coeff1, in1, 0));
#else
  out = in0 + in1;
#endif

#elif ELTWISE_TYPE == 2
  out = fmax(in0, in1);
#elif ELTWISE_TYPE == 3
  out = fmin(in0, in1);
#endif

  WRITE_IMAGET(output, (int2)(w, hb), out);
}

