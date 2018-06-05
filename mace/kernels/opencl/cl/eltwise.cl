#include <common.h>

__kernel void eltwise(KERNEL_ERROR_PARAMS
                      GLOBAL_WORK_GROUP_SIZE_DIM3
                      __read_only image2d_t input0,
#if INPUT_TYPE == 1
                      __private const float value,
#else
                      __read_only image2d_t input1,
#endif
                      __private const int height,
                      __private const int width,
                      __private const int channel,
#ifdef COEFF_SUM
                      __private const float coeff0,
                      __private const float coeff1,
#endif
                      __write_only image2d_t output) {
  const int chan_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (chan_idx >= global_size_dim0 ||
      width_idx >= global_size_dim1 || hb >= global_size_dim2)
    return;
#endif

  const int pos = mad24(chan_idx, width, width_idx);
  DATA_TYPE4 in0 = READ_IMAGET(input0, SAMPLER, (int2)(pos, hb));
#if INPUT_TYPE == 1
  DATA_TYPE4 in1 = (DATA_TYPE4)(value, value, value, value);
#elif INPUT_TYPE == 2
  const int batch_idx = hb / height;
  DATA_TYPE4 in1 = READ_IMAGET(input1, SAMPLER, (int2)(chan_idx, batch_idx));
#elif INPUT_TYPE == 3
  DATA_TYPE4 in1 = READ_IMAGET(input1, SAMPLER, (int2)(chan_idx, 0));
#else
  DATA_TYPE4 in1 = READ_IMAGET(input1, SAMPLER, (int2)(pos, hb));
#endif

  DATA_TYPE4 out;
#if ELTWISE_TYPE == 0
  #ifdef COEFF_SUM
    out = mad(coeff1, in0, mad(coeff0, in1, 0));
  #else
    out = in0 + in1;
  #endif
#elif ELTWISE_TYPE == 1
  #ifdef SWAPPED
    out = in1 - in0;
  #else
    out = in0 - in1;
  #endif
#elif ELTWISE_TYPE == 2
  out = in0 * in1;
#elif ELTWISE_TYPE == 3
  #ifdef SWAPPED
    out = in1 / in0;
  #else
    out = in0 / in1;
  #endif
#elif ELTWISE_TYPE == 4
  out = fmin(in0, in1);
#elif ELTWISE_TYPE == 5
  out = fmax(in0, in1);
#elif ELTWISE_TYPE == 6
  in1 = (DATA_TYPE4)(0, 0, 0, 0);
  out = in1 - in0;
#elif ELTWISE_TYPE == 7
  out = fabs(in0);
#elif ELTWISE_TYPE == 8
  DATA_TYPE4 diff = in0 - in1;
  out = diff * diff;
#elif ELTWISE_TYPE == 9
  #ifdef SWAPPED
    out = pow(in1, in0);
  #else
    out = pow(in0, in1);
  #endif
#endif

#if INPUT_TYPE == 1
  #if ELTWISE_TYPE == 0 || ELTWISE_TYPE == 1 || ELTWISE_TYPE == 4 ||          \
      ELTWISE_TYPE == 5 || ELTWISE_TYPE == 8 || ELTWISE_TYPE == 9
    const int remain_channel = channel - 4 * chan_idx;
    if (remain_channel < 4) {
      switch (remain_channel) {
        case 1:
          out.y = 0;
        case 2:
          out.z = 0;
        case 3:
          out.w = 0;
      }
    }
  #endif
#endif

  WRITE_IMAGET(output, (int2)(pos, hb), out);
}
