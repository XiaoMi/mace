// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_OPS_ARM_CONV_WINOGRAD_H_
#define MACE_OPS_ARM_CONV_WINOGRAD_H_

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "mace/core/types.h"
#include "mace/ops/sgemm.h"

namespace mace {
namespace ops {

void TransformFilter4x4(const float *filter,
                        const index_t in_channels,
                        const index_t out_channels,
                        float *output);

void TransformFilter8x8(const float *filter,
                        const index_t in_channels,
                        const index_t out_channels,
                        float *output);

void WinogradConv3x3s1(const float *input,
                       const float *filter,
                       const index_t batch,
                       const index_t in_height,
                       const index_t in_width,
                       const index_t in_channels,
                       const index_t out_channels,
                       const int out_tile_size,
                       float *output,
                       SGemm *sgemm,
                       ScratchBuffer *scratch_buffer);

void WinogradConv3x3s1(const float *input,
                       const float *transformed_filter,
                       const index_t batch,
                       const index_t in_height,
                       const index_t in_width,
                       const index_t in_channels,
                       const index_t out_channels,
                       const int out_tile_size,
                       float *transformed_input,
                       float *transformed_output,
                       float *output,
                       SGemm *sgemm,
                       ScratchBuffer *scratch_buffer);

void ConvRef3x3s1(const float *input,
                  const float *filter,
                  const index_t batch,
                  const index_t in_height,
                  const index_t in_width,
                  const index_t in_channels,
                  const index_t out_channels,
                  float *output);

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_CONV_WINOGRAD_H_
