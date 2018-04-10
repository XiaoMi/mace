//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_GEMM_H_
#define MACE_KERNELS_GEMM_H_

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "mace/core/types.h"

namespace mace {
namespace kernels {

void Gemm(const float *A,
          const float *B,
          const index_t batch,
          const index_t height,
          const index_t K,
          const index_t width,
          float *C);

void GemmRef(const float *A,
             const float *B,
             const index_t height,
             const index_t K,
             const index_t width,
             float *C);

void Gemv(const float *m_ptr,
          const float *v_ptr,
          const index_t width,
          const index_t height,
          float *out_ptr);

void GemvRef(const float *m_ptr,
             const float *v_ptr,
             const index_t width,
             const index_t height,
             float *out_ptr);

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_GEMM_H_
