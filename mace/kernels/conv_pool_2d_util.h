// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

void CalcPaddingAndOutputSize(const index_t *input_shape,
                              const DataFormat input_format,
                              const index_t *filter_shape,
                              const DataFormat filter_format,
                              const int *dilations,
                              const int *strides,
                              Padding padding,
                              index_t *output_shape,
                              int *padding_size);

void CalcNCHWPaddingAndOutputSize(const index_t *input_shape,
                                  const index_t *filter_shape,
                                  const int *dilations,
                                  const int *strides,
                                  Padding padding,
                                  index_t *output_shape,
                                  int *padding_size);

void CalcNHWCPaddingAndOutputSize(const index_t *input_shape,
                                  const index_t *filter_shape,
                                  const int *dilations,
                                  const int *strides,
                                  Padding padding,
                                  index_t *output_shape,
                                  int *padding_size);

void CalcOutputSize(const index_t *input_shape,
                    const DataFormat input_format,
                    const index_t *filter_shape,
                    const DataFormat filter_format,
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape);

void CalcOutputSize(const index_t *input_shape,   // NHWC
                    const index_t *filter_shape,  // OIHW
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape);

void CalcNCHWOutputSize(const index_t *input_shape,
                    const index_t *filter_shape,
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

MaceStatus ConstructNCHWInputWithSpecificPadding(const Tensor *input,
                               const int pad_top, const int pad_bottom,
                               const int pad_left, const int pad_right,
                               Tensor *output_tensor);

MaceStatus ConstructNCHWInputWithPadding(const Tensor *input,
                                   const int *paddings,
                                   Tensor *output_tensor,
                                   bool padding_same_value = false);

MaceStatus ConstructNHWCInputWithPadding(const Tensor *input,
                                   const int *paddings,
                                   Tensor *output_tensor,
                                   bool padding_same_value = false);

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_CONV_POOL_2D_UTIL_H_
