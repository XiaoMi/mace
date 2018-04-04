#include <common.h>

__kernel void cwise(KERNEL_ERROR_PARAMS
                    GLOBAL_WORK_GROUP_SIZE_DIM2
                    __read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
                    __private const float value,
                    __write_only image2d_t output) {
  const int w = get_global_id(0);
  const int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || hb >= global_size_dim1) return;
#endif

  DATA_TYPE4 in0 = READ_IMAGET(input, SAMPLER, (int2)(w, hb));
  DATA_TYPE4 in1 = (DATA_TYPE4){value, value, value, value};
  DATA_TYPE4 out;

#if CWISE_TYPE == 0
  out = in0 * in1;
#elif CWISE_TYPE == 1
  out = in0 + in1;
#elif CWISE_TYPE == 2
  out.x = fmax(in0.x, value);
  out.y = fmax(in0.y, value);
  out.z = fmax(in0.z, value);
  out.z = fmax(in0.w, value);
#elif CWISE_TYPE == 3
  out.x = fmin(in0.x, value);
  out.y = fmin(in0.y, value);
  out.z = fmin(in0.z, value);
  out.z = fmin(in0.w, value);
#elif CWISE_TYPE == 4
  out = in0 - in1;
#elif CWISE_TYPE == 5
  out = in0 / in1;
#elif CWISE_TYPE == 6
  in1 = (DATA_TYPE4)(0, 0, 0, 0);
  out = in1 - in0;
#elif CWISE_TYPE == 7
  out.x = fabs(in0.x);
  out.y = fabs(in0.y);
  out.z = fabs(in0.z);
  out.w = fabs(in0.w);
#endif

  check_out_of_range_for_image2d(output, w, hb, kernel_error);

  WRITE_IMAGET(output, (int2)(w, hb), out);
}
