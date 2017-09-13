//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_NEON_CONV_2D_NEON_BASE_H_
#define MACE_KERNELS_NEON_CONV_2D_NEON_BASE_H_

#include "mace/core/common.h"

namespace mace {
namespace kernels {

template <index_t kernel_h, index_t kernel_w, index_t stride_h, index_t stride_w>
inline void Conv2dNeon(const float* input, // NCHW
                           const index_t* input_shape,
                           const float* filter, // c_out, c_in, kernel_h, kernel_w
                           const float* bias, // c_out
                           float* output, // NCHW
                           const index_t* output_shape);

} //  namespace kernels
} //  namespace mace

#endif //  MACE_KERNELS_NEON_CONV_2D_NEON_BASE_H_
