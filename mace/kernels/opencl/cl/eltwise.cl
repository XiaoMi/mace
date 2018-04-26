#include <common.h>

__kernel void eltwise(KERNEL_ERROR_PARAMS
                      GLOBAL_WORK_GROUP_SIZE_DIM3
                      __read_only image2d_t input0,
                      __read_only image2d_t input1,
                      __private const float value,
                      __private const int height,
                      __private const int width,
                      __private const int channel,
#ifdef COEFF_SUM
                      __private const float coeff0,
                      __private const float coeff1,
#endif
                      __write_only image2d_t output) {
  const int c = get_global_id(0);
  const int w = get_global_id(1);
  const int hb = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (c >= global_size_dim0 || w >= global_size_dim1 || hb >= global_size_dim2)
    return;
#endif

  int pos_w;
  int pos_h;
#if START_AXIS == 0
  pos_w = mad24(c, width, w);
  pos_h = hb;
#elif START_AXIS == 1
  pos_w = mad24(c, width, w);
  pos_h = hb % height;
#elif START_AXIS == 2
  pos_w = mad24(c, width, w);
  pos_h = 0;
#elif START_AXIS == 3
  pos_w = c;
  pos_h = 0;
#endif
  const int pos = mad24(c, width, w);
  const int remain_channel = channel - 4 * c;
  DATA_TYPE4 in0 = READ_IMAGET(input0, SAMPLER, (int2)(pos, hb));
  DATA_TYPE4 in1 ;
#if IS_SCALER == 1
  in1 = (DATA_TYPE4){value, value, value, value};
#else
  in1 = READ_IMAGET(input1, SAMPLER, (int2)(pos_w, pos_h));
#endif
  DATA_TYPE4 out;
#if ELTWISE_TYPE == 0
  out = in0 * in1;
#elif ELTWISE_TYPE == 1

#ifdef COEFF_SUM
  #if NEEDSWAP == 0
    out = mad(coeff0, in0, mad(coeff1, in1, 0));
  #else
    out = mad(coeff1, in0, mad(coeff0, in1, 0));
  #endif
#else
  out = in0 + in1;
#endif

#elif ELTWISE_TYPE == 2
  out = fmax(in0, in1);
#elif ELTWISE_TYPE == 3
  out = fmin(in0, in1);
#elif ELTWISE_TYPE == 4
  #if NEED_SWAP == 0
    out = in0 - in1;
  #else
    out = in1 - in0;
  #endif
#elif ELTWISE_TYPE == 5
  #if NEED_SWAP == 0
    if (fabs(in1.x) > 0.000001f)
      out.x = in0.x / in1.x;
    if (fabs(in1.y) > 0.000001f)
      out.y = in0.y / in1.y;
    if (fabs(in1.z) > 0.000001f)
      out.z = in0.z / in1.z;
    if (fabs(in1.w) > 0.000001f)
      out.w = in0.w / in1.w;
  #else
    if (fabs(in1.x) > 0.000001f)
      out.x = in1.x / in0.x;
    if (fabs(in1.y) > 0.000001f)
      out.y = in1.y / in0.y;
    if (fabs(in1.z) > 0.000001f)
      out.z = in1.z / in0.z;
    if (fabs(in1.w) > 0.000001f)
      out.w = in1.w / in0.w;
  #endif
#elif ELTWISE_TYPE == 8
  DATA_TYPE4 diff = in0 - in1;
  out = diff * diff;
#endif

#if ELTWISE_TYPE == 1 || ELTWISE_TYPE == 2 || ELTWISE_TYPE == 3 \
   || ELTWISE_TYPE == 4 || ELTWISE_TYPE == 8
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

  WRITE_IMAGET(output, (int2)(pos, hb), out);
}
