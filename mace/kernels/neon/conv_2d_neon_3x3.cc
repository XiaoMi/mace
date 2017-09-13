//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <arm_neon.h>
#include "mace/kernels/neon/conv_2d_neon_base.h"

namespace mace {
namespace kernels {

template<>
void Conv2dNeon<3, 3, 1, 1>(const float* input, // NCHW
                       const index_t* input_shape,
                       const float* filter, // c_out, c_in, kernel_h, kernel_w
                       const float* bias, // c_out
                       float* output, // NCHW
                       const index_t* output_shape) {

}

} //  namespace kernels
} //  namespace mace
