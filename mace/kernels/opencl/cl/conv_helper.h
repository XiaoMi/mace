//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_OPENCL_CL_CONV_HELPER_H_
#define MACE_KERNELS_OPENCL_CL_CONV_HELPER_H_

float4 conv1x3_s1(const float *input_ptr,
                  const float *filter_ptr);
float4 conv1x3_s2(const float *input_ptr,
                  const float *filter_ptr);
float conv3x3(const float *input_ptr,
              const float *filter_ptr,
              const int row_width);
#endif //  MACE_KERNELS_OPENCL_CL_CONV_HELPER_H_
