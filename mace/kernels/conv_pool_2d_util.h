//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_CONV_POOL_2D_UTIL_H_
#define MACE_KERNELS_CONV_POOL_2D_UTIL_H_

#include "mace/core/tensor.h"

namespace mace {

enum Padding {
  VALID = 0,  // No padding
  SAME = 1,   // Pads with half the filter size (rounded down) on both sides
  FULL = 2,   // Pads with one less than the filter size on both sides
};

namespace kernels {

void CalcPaddingAndOutputSize(const index_t *input_shape,   // NCHW
                              const index_t *filter_shape,  // OIHW
                              const int *dilations, const int *strides,
                              Padding padding, index_t *output_shape,
                              int *padding_size);

void ConstructInputWithPadding(const float *input, const index_t *input_shape,
                               const int *paddings, Tensor *output_tensor);
}  //  namespace kernels
}  //  namespace mace

#endif  // MACE_KERNELS_CONV_POOL_2D_UTIL_H_
