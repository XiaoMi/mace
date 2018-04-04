//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_OPENCL_CL_COMMON_H_
#define MACE_KERNELS_OPENCL_CL_COMMON_H_

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define VEC_DATA_TYPE_STR(data_type, size) data_type##size
#define VEC_DATA_TYPE(data_type, size) VEC_DATA_TYPE_STR(data_type, size)

#define CMD_TYPE_STR(cmd, type) cmd##type
#define CMD_TYPE(cmd, type) CMD_TYPE_STR(cmd, type)

#define DATA_TYPE4 VEC_DATA_TYPE(DATA_TYPE, 4)
#define READ_IMAGET CMD_TYPE(read_image, CMD_DATA_TYPE)
#define WRITE_IMAGET CMD_TYPE(write_image, CMD_DATA_TYPE)


#ifndef NON_UNIFORM_WORK_GROUP

#define GLOBAL_WORK_GROUP_SIZE_DIM2 \
    __private const int global_size_dim0,       \
    __private const int global_size_dim1,
#define GLOBAL_WORK_GROUP_SIZE_DIM3 \
    __private const int global_size_dim0,       \
    __private const int global_size_dim1,       \
    __private const int global_size_dim2,

#else

#define GLOBAL_WORK_GROUP_SIZE_DIM2
#define GLOBAL_WORK_GROUP_SIZE_DIM3

#endif


#ifdef OUT_OF_RANGE_CHECK

#define KERNEL_ERROR_PARAMS \
  __global char *kernel_error,

#else

#define KERNEL_ERROR_PARAMS

#endif

#ifdef OUT_OF_RANGE_CHECK
#define CHECK_OUT_OF_RANGE_FOR_IMAGE2D(image, x, y, kernel_error) \
  check_out_of_range_for_image2d(image, x, y, kernel_error)
#else
#define CHECK_OUT_OF_RANGE_FOR_IMAGE2D(image, x, y, kernel_error)
#endif

__constant sampler_t SAMPLER =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

inline DATA_TYPE4 do_activation(DATA_TYPE4 in,
#ifdef USE_PRELU
                                DATA_TYPE4 prelu_alpha,
#endif
                                __private const float relux_max_limit) {
  DATA_TYPE4 out;
#ifdef USE_RELU
  out = fmax(in, (DATA_TYPE)0);
#endif
#ifdef USE_RELUX
  out = clamp(in, (DATA_TYPE4)0, relux_max_limit);
#endif
#ifdef USE_PRELU
  out = select(prelu_alpha * in, in, in >= (DATA_TYPE)0);
#endif
#ifdef USE_TANH
  out = tanh(in);
#endif
#ifdef USE_SIGMOID
  out = native_recip((DATA_TYPE)1 + native_exp(-in));
#endif
  return out;
}

inline void check_out_of_range_for_image2d(__write_only image2d_t image,
                                           __private const int x,
                                           __private const int y,
                                           global char *kernel_error) {
#ifdef OUT_OF_RANGE_CHECK
  int2 image_dim = get_image_dim(image);
  if (x >= image_dim.x || y >= image_dim.y) {
    *kernel_error = '1';
  }
#endif
}

#endif  // MACE_KERNELS_OPENCL_CL_COMMON_H_
