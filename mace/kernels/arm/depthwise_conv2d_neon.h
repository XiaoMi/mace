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

#ifndef MACE_KERNELS_ARM_DEPTHWISE_CONV2D_NEON_H_
#define MACE_KERNELS_ARM_DEPTHWISE_CONV2D_NEON_H_

#include "mace/core/types.h"

namespace mace {
namespace kernels {

void DepthwiseConv2dNeonK3x3S1(const float *input,
                               const float *filter,
                               const index_t *in_shape,
                               const index_t *out_shape,
                               const int *pad_hw,
                               const index_t valid_h_start,
                               const index_t valid_h_stop,
                               const index_t valid_w_start,
                               const index_t valid_w_stop,
                               float *output);

void DepthwiseConv2dNeonK3x3S2(const float *input,
                               const float *filter,
                               const index_t *in_shape,
                               const index_t *out_shape,
                               const int *pad_hw,
                               const index_t valid_h_start,
                               const index_t valid_h_stop,
                               const index_t valid_w_start,
                               const index_t valid_w_stop,
                               float *output);

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_ARM_DEPTHWISE_CONV2D_NEON_H_
