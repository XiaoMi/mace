//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#ifndef MACE_KERNELS_NEON_CONV_2D_NEON_3X3_H_
#define MACE_KERNELS_NEON_CONV_2D_NEON_3X3_H_

#include <arm_neon.h>
#include "mace/core/common.h"

namespace mace {
namespace kernels {

void Conv2dNeonK3x3S1(const float* input, // NCHW
                       const index_t* input_shape,
                       const float* filter, // c_out, c_in, kernel_h, kernel_w
                       const float* bias, // c_out
                       float* output, // NCHW
                       const index_t* output_shape) {

}

} //  namespace kernels
} //  namespace mace

#endif //  MACE_KERNELS_NEON_CONV_2D_NEON_3X3_H_
