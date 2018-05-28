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

#ifndef MACE_KERNELS_ARM_CONV_2D_NEON_H_
#define MACE_KERNELS_ARM_CONV_2D_NEON_H_

#include "mace/core/types.h"

namespace mace {
namespace kernels {

void Conv2dNeonK1x1S1(const float *input,
                      const float *filter,
                      const index_t batch,
                      const index_t height,
                      const index_t width,
                      const index_t in_channels,
                      const index_t out_channels,
                      float *output);

void Conv2dNeonK3x3S1(const float *input,
                      const float *filter,
                      const index_t *in_shape,
                      const index_t *out_shape,
                      float *output);

void Conv2dNeonK3x3S2(const float *input,
                      const float *filter,
                      const index_t *in_shape,
                      const index_t *out_shape,
                      float *output);

void Conv2dNeonK5x5S1(const float *input,
                      const float *filter,
                      const index_t *in_shape,
                      const index_t *out_shape,
                      float *output);

void Conv2dNeonK1x7S1(const float *input,
                      const float *filter,
                      const index_t *in_shape,
                      const index_t *out_shape,
                      float *output);

void Conv2dNeonK7x1S1(const float *input,
                      const float *filter,
                      const index_t *in_shape,
                      const index_t *out_shape,
                      float *output);

void Conv2dNeonK7x7S1(const float *input,
                      const float *filter,
                      const index_t *in_shape,
                      const index_t *out_shape,
                      float *output);

void Conv2dNeonK7x7S2(const float *input,
                      const float *filter,
                      const index_t *in_shape,
                      const index_t *out_shape,
                      float *output);

void Conv2dNeonK7x7S3(const float *input,
                      const float *filter,
                      const index_t *in_shape,
                      const index_t *out_shape,
                      float *output);

void Conv2dNeonK1x15S1(const float *input,
                       const float *filter,
                       const index_t *in_shape,
                       const index_t *out_shape,
                       float *output);

void Conv2dNeonK15x1S1(const float *input,
                       const float *filter,
                       const index_t *in_shape,
                       const index_t *out_shape,
                       float *output);

// calculate one output channel and one input channel
inline void Conv2dCPUKHxKWCalc(const float *in_ptr,
                               const float *filter_ptr,
                               const index_t in_width,
                               const index_t filter_height,
                               const index_t filter_width,
                               const index_t out_height,
                               const index_t out_width,
                               float *out_ptr,
                               const int stride) {
  for (index_t h = 0; h < out_height; ++h) {
    for (index_t w = 0; w < out_width; ++w) {
      for (int i = 0; i < filter_height; ++i) {
        for (int j = 0; j < filter_width; ++j) {
          out_ptr[h * out_width + w] +=
              in_ptr[(h * stride + i) * in_width + (w * stride + j)] *
              filter_ptr[i * filter_width + j];
        }
      }
    }
  }
}

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_ARM_CONV_2D_NEON_H_
