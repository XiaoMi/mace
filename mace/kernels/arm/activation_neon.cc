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

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#include <algorithm>
#include "mace/kernels/arm/activation_neon.h"

namespace mace {
namespace kernels {

void ReluNeon(const float *input, const index_t size, float *output) {
#if defined(MACE_ENABLE_NEON)
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for
  for (index_t i = 0; i <= size - 4; i += 4) {
    float32x4_t v = vld1q_f32(input + i);
    v = vmaxq_f32(v, vzero);
    vst1q_f32(output + i, v);
  }
  // remain
  for (index_t i = (size >> 2) << 2; i < size; ++i) {
    output[i] = std::max(input[i], 0.f);
  }
#else
#pragma omp parallel for
  for (index_t i = 0; i < size; ++i) {
    output[i] = std::max(input[i], 0.f);
  }
#endif
}

void ReluxNeon(const float *input, const float limit,
               const index_t size, float *output) {
#if defined(MACE_ENABLE_NEON)
  float32x4_t vzero = vdupq_n_f32(0.f);
  float32x4_t vlimit = vdupq_n_f32(limit);
#pragma omp parallel for
  for (index_t i = 0; i <= size - 4; i += 4) {
    float32x4_t v = vld1q_f32(input + i);
    v = vmaxq_f32(v, vzero);
    v = vminq_f32(v, vlimit);
    vst1q_f32(output + i, v);
  }
  // remain
  for (index_t i = (size >> 2) << 2; i < size; ++i) {
    output[i] = std::min(std::max(input[i], 0.f), limit);
  }
#else
#pragma omp parallel for
  for (index_t i = 0; i < size; ++i) {
    output[i] = std::min(std::max(input[i], 0.f), limit);
  }
#endif
}

}  // namespace kernels
}  // namespace mace
