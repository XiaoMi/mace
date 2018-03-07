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

enum RoundType {
  FLOOR = 0,
  CEIL = 1,
};

namespace kernels {

void CalcPaddingAndOutputSize(const index_t *input_shape,   // NCHW
                              const index_t *filter_shape,  // OIHW
                              const int *dilations,
                              const int *strides,
                              Padding padding,
                              index_t *output_shape,
                              int *padding_size);

void CalcNHWCPaddingAndOutputSize(const index_t *input_shape,   // NCHW
                                  const index_t *filter_shape,  // OIHW
                                  const int *dilations,
                                  const int *strides,
                                  Padding padding,
                                  index_t *output_shape,
                                  int *padding_size);

void CalcOutputSize(const index_t *input_shape,   // NHWC
                    const index_t *filter_shape,  // HWOI
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape);

void CalPaddingSize(const index_t *input_shape,   // NCHW
                    const index_t *filter_shape,  // OIHW
                    const int *dilations,
                    const int *strides,
                    Padding padding,
                    int *padding_size);

void ConstructInputWithPadding(const Tensor *input,
                               const int *paddings,
                               Tensor *output_tensor,
                               bool padding_same_value = false);

void ConstructNHWCInputWithPadding(const Tensor *input,
                                   const int *paddings,
                                   Tensor *output_tensor,
                                   bool padding_same_value = false);

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_CONV_POOL_2D_UTIL_H_
