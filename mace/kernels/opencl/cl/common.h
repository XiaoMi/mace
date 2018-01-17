//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_OPENCL_CL_COMMON_H_
#define MACE_KERNELS_OPENCL_CL_COMMON_H_

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

#define VEC_DATA_TYPE_STR(data_type, size) data_type##size
#define VEC_DATA_TYPE(data_type, size) VEC_DATA_TYPE_STR(data_type, size)

#define CMD_TYPE_STR(cmd, type) cmd##type
#define CMD_TYPE(cmd, type) CMD_TYPE_STR(cmd, type)

#define DATA_TYPE4 VEC_DATA_TYPE(DATA_TYPE, 4)
#define READ_IMAGET CMD_TYPE(read_image, CMD_DATA_TYPE)
#define WRITE_IMAGET CMD_TYPE(write_image, CMD_DATA_TYPE)

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;


inline DATA_TYPE4 do_activation(DATA_TYPE4 in,
                                __private const DATA_TYPE relux_max_limit,
                                __private const DATA_TYPE prelu_alpha) {
  DATA_TYPE4 out;
#ifdef USE_RELU
  out = fmax(in, 0);
#endif
#ifdef USE_RELUX
  out = clamp(in, 0, relux_max_limit);
#endif
#ifdef USE_PRELU
  out = select(prelu_alpha * in, in, in >= 0);
#endif
#ifdef USE_TANH
  out = tanh(in);
#endif
#ifdef USE_SIGMOID
  out = native_recip(1.0 + native_exp(-in));
#endif
  return out;
}

#endif  // MACE_KERNELS_OPENCL_CL_COMMON_H_
