#include <common.h>

__kernel void cwise(KERNEL_ERROR_PARAMS
                    GLOBAL_WORK_GROUP_SIZE_DIM2
                    __read_only image2d_t input, /* [c%4 * w * c/4, h * b] */
                    __private const int width,
                    __private const int channel,
                    __private const float value,
                    __write_only image2d_t output) {
  const int w = get_global_id(0);
  const int hb = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (w >= global_size_dim0 || hb >= global_size_dim1) return;
#endif

  const int remain_chan = channel - mul24((w / width), 4);

  DATA_TYPE4 in0 = READ_IMAGET(input, SAMPLER, (int2)(w, hb));
  DATA_TYPE4 in1 = (DATA_TYPE4){value, value, value, value};
  DATA_TYPE4 out;

#if CWISE_TYPE == 0
  out = in0 * in1;
#elif CWISE_TYPE == 1
  out = in0 + in1;
#elif CWISE_TYPE == 2
  out = fmax(in0, in1);
#elif CWISE_TYPE == 3
  out = fmin(in0, in1);
#elif CWISE_TYPE == 4
  out = in0 - in1;
#elif CWISE_TYPE == 5
  out = in0 / in1;
#elif CWISE_TYPE == 6
  in1 = (DATA_TYPE4)(0, 0, 0, 0);
  out = in1 - in0;
#elif CWISE_TYPE == 7
  out = fabs(in0);
#endif

#if CWISE_TYPE == 1 || CWISE_TYPE == 2 || CWISE_TYPE == 3 || CWISE_TYPE == 4
  if (remain_chan < 4) {
    switch (remain_chan) {
      case 1:
        out.y = 0;
      case 2:
        out.z = 0;
      case 3:
        out.w = 0;
    }
  }
#endif

  WRITE_IMAGET(output, (int2)(w, hb), out);
}
