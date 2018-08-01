#include <common.h>
// Supported data types: half/float
__kernel void batch_norm(KERNEL_ERROR_PARAMS
                         GLOBAL_WORK_GROUP_SIZE_DIM3
                         __read_only image2d_t input,
                         __read_only image2d_t scale,
                         __read_only image2d_t offset,
#ifndef FOLDED_CONSTANT
                         __read_only image2d_t mean,
                         __read_only image2d_t var,
                         __private const float epsilon,
#endif
                         __write_only image2d_t output,
                         __private const float relux_max_limit) {
  const int ch_blk = get_global_id(0);
  const int w = get_global_id(1);
  const int hb = get_global_id(2);

#ifndef NON_UNIFORM_WORK_GROUP
  if (ch_blk >= global_size_dim0 || w >= global_size_dim1
      || hb >= global_size_dim2) {
    return;
  }
#endif
  const int width = global_size_dim1;

#ifdef FOLDED_CONSTANT
  DATA_TYPE4 bn_scale = READ_IMAGET(scale, SAMPLER, (int2)(ch_blk, 0));
  DATA_TYPE4 bn_offset = READ_IMAGET(offset, SAMPLER, (int2)(ch_blk, 0));
#else
  DATA_TYPE4 scale_value = READ_IMAGET(scale, SAMPLER, (int2)(ch_blk, 0));
  DATA_TYPE4 offset_value = READ_IMAGET(offset, SAMPLER, (int2)(ch_blk, 0));
  DATA_TYPE4 mean_value = READ_IMAGET(mean, SAMPLER, (int2)(ch_blk, 0));
  DATA_TYPE4 var_value = READ_IMAGET(var, SAMPLER, (int2)(ch_blk, 0));

  // native_rsqrt seems not faster than rsqrt
  DATA_TYPE4 bn_scale = scale_value * rsqrt(var_value + (DATA_TYPE4)epsilon);
  DATA_TYPE4 bn_offset = mad(0 - mean_value, bn_scale, offset_value);
#endif

  const int pos = mad24(ch_blk, width, w);

  DATA_TYPE4 in = READ_IMAGET(input, SAMPLER, (int2)(pos, hb));
  DATA_TYPE4 out = mad(in, bn_scale, bn_offset);

#if defined(USE_RELU) || defined(USE_RELUX) || defined(USE_TANH) || defined(USE_SIGMOID)
  out = do_activation(out, relux_max_limit);
#endif

  WRITE_IMAGET(output, (int2)(pos, hb), out);
}
