#include <common.h>

__kernel void scalar_math(__read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
                      __private const float scalar,
                      __write_only image2d_t output) {
  const int w = get_global_id(0);
  const int hb = get_global_id(1);

  DATA_TYPE4 in0 = READ_IMAGET(input, SAMPLER, (int2)(w, hb));
  DATA_TYPE4 in1;
  in1.x = scalar;
  in1.y = scalar;
  in1.z = scalar;
  in1.w = scalar;  
  DATA_TYPE4 out;
#if SCALAR_MATH_TYPE == 1
  out = in0 + in1;
#elif SCALAR_MATH_TYPE == 4
  out = in0 - in1;
#elif SCALAR_MATH_TYPE == 0
  out = in0 * in1;
#elif SCALAR_MATH_TYPE == 5
  out = in0 / in1;
#endif

  WRITE_IMAGET(output, (int2)(w, hb), out);
}
