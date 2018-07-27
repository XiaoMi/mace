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

#ifndef MACE_KERNELS_GEMM_H_
#define MACE_KERNELS_GEMM_H_

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "mace/core/types.h"

// Gemm function does fast matrix-matrix multiplications with batch.
// Gemv function does fast matrix-vector multiplications with batch.

namespace mace {
namespace kernels {

// Gemm calculates A[batch, height, K] dot B[batch, K, width] within each batch,
// and output to C[batch, height, width].
// height, K, width correspond to matrix dimension size after transpose (if any)
void Gemm(const float *A,
          const float *B,
          const index_t batch,
          const index_t height,
          const index_t K,
          const index_t width,
          float *C,
          const bool transpose_a = false,
          const bool transpose_b = false);

void GemmRef(const float *A,
             const float *B,
             const index_t batch,
             const index_t height,
             const index_t K,
             const index_t width,
             float *C,
             const bool transpose_a = false,
             const bool transpose_b = false);

// Gemm calculates M[height, width] dot V[batch, height] within each batch of V,
// and output to out[batch, width].
void Gemv(const float *m_ptr,
          const float *v_ptr,
          const index_t batch,
          const index_t width,
          const index_t height,
          float *out_ptr);

void GemvRef(const float *m_ptr,
             const float *v_ptr,
             const index_t batch,
             const index_t width,
             const index_t height,
             float *out_ptr);

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_GEMM_H_
