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

#endif  // MACE_KERNELS_OPENCL_CL_COMMON_H_
