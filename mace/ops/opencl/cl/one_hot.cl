#include <common.h>

__kernel void one_hot(OUT_OF_RANGE_PARAMS
                      GLOBAL_WORK_GROUP_SIZE_DIM2
                      __read_only image2d_t input,
                      __write_only image2d_t output,
#ifdef AXIS_0
                      __private const int in_size,
#endif
                      __private const float on_value,
                      __private const float off_value) {

  const int channel_idx = get_global_id(0);
  const int batch_idx   = get_global_id(1);

#ifndef NON_UNIFORM_WORK_GROUP
  if (channel_idx >= global_size_dim0 || batch_idx >= global_size_dim1) {
    return;
  }
#endif

  DATA_TYPE4 out = off_value;

#ifdef AXIS_0
  int in_idx = channel_idx * 4;
  DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(0, in_idx));

  if (in.s0 == batch_idx) {
    out.s0 = on_value;
  }

  if (++in_idx < in_size) {
    in = READ_IMAGET(input, SAMPLER, (int2)(0, in_idx));

    if (in.s0 == batch_idx) {
      out.s1 = on_value;
    }

    if (++in_idx < in_size) {
      in = READ_IMAGET(input, SAMPLER, (int2)(0, in_idx));

      if (in.s0 == batch_idx) {
        out.s2 = on_value;
      }

      if (++in_idx < in_size) {
        in = READ_IMAGET(input, SAMPLER, (int2)(0, in_idx));

        if (in.s0 == batch_idx) {
          out.s3 = on_value;
        }
      }
    }
  }
#else
  DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(0, batch_idx));
  int i = in.s0;

  if (i / 4 == channel_idx) {
    switch (i % 4) {
    case 0:
      out.s0 = on_value;
      break;
    case 1:
      out.s1 = on_value;
      break;
    case 2:
      out.s2 = on_value;
      break;
    case 3:
      out.s3 = on_value;
      break;
    }
  }
#endif

  WRITE_IMAGET(output, (int2)(channel_idx, batch_idx), out);
}
