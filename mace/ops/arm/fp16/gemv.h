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

#ifndef MACE_OPS_ARM_FP16_GEMV_H_
#define MACE_OPS_ARM_FP16_GEMV_H_

#include "mace/core/types.h"

#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
#include <arm_neon.h>
#endif

#if defined(MACE_ENABLE_NEON) && !defined(__aarch64__) && defined(__ANDROID__)
#define vaddvq_f32(v) ((v)[0] + (v)[1] + (v)[2] + (v)[3])
#endif

namespace mace {
namespace ops {

template<typename INPUT_TYPE_LEFT,
         typename INPUT_TYPE_RIGHT,
         typename OUTPUT_TYPE>
void FP16Gemv(const INPUT_TYPE_LEFT *m_ptr,
              const INPUT_TYPE_RIGHT *v_ptr,
              const index_t height,
              const index_t width,
              OUTPUT_TYPE *result);

#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
template<>
void FP16Gemv<float16_t, float, float>(const float16_t *m_ptr,
                                       const float *v_ptr,
                                       const index_t height,
                                       const index_t width,
                                       float *out_ptr) {
#pragma omp parallel for
  for (index_t h = 0; h < height; ++h) {
    const float16_t *m_ptr0 = m_ptr + h * width;
    const float *v_ptr0 = v_ptr;
    float *out_ptr0 = out_ptr + h;
    float sum0 = 0;

    float32x4_t vm0, vm1, vm2, vm3;
    float32x4_t vv0, vv1, vv2, vv3;
    float32x4_t vsum0 = vdupq_n_f32(0.f);
    float32x4_t vsum1 = vdupq_n_f32(0.f);
    float32x4_t vsum2 = vdupq_n_f32(0.f);
    float32x4_t vsum3 = vdupq_n_f32(0.f);

    index_t w;
    for (w = 0; w + 15 < width; w += 16) {
      vm0 = vcvt_f32_f16(vld1_f16(m_ptr0));
      vv0 = vld1q_f32(v_ptr0);
      vm1 = vcvt_f32_f16(vld1_f16(m_ptr0 + 4));
      vv1 = vld1q_f32(v_ptr0 + 4);
      vm2 = vcvt_f32_f16(vld1_f16(m_ptr0 + 8));
      vv2 = vld1q_f32(v_ptr0 + 8);
      vm3 = vcvt_f32_f16(vld1_f16(m_ptr0 + 12));
      vv3 = vld1q_f32(v_ptr0 + 12);

      vsum0 = vmlaq_f32(vsum0, vm0, vv0);
      vsum1 = vmlaq_f32(vsum1, vm1, vv1);
      vsum2 = vmlaq_f32(vsum2, vm2, vv2);
      vsum3 = vmlaq_f32(vsum3, vm3, vv3);

      m_ptr0 += 16;
      v_ptr0 += 16;
    }

    for (; w + 7 < width; w += 8) {
      vm0 = vcvt_f32_f16(vld1_f16(m_ptr0));
      vv0 = vld1q_f32(v_ptr0);
      vm1 = vcvt_f32_f16(vld1_f16(m_ptr0 + 4));
      vv1 = vld1q_f32(v_ptr0 + 4);

      vsum0 = vmlaq_f32(vsum0, vm0, vv0);
      vsum1 = vmlaq_f32(vsum1, vm1, vv1);

      m_ptr0 += 8;
      v_ptr0 += 8;
    }

    for (; w + 3 < width; w += 4) {
      vm0 = vcvt_f32_f16(vld1_f16(m_ptr0));
      vv0 = vld1q_f32(v_ptr0);
      vsum0 = vmlaq_f32(vsum0, vm0, vv0);

      m_ptr0 += 4;
      v_ptr0 += 4;
    }
    vsum0 += vsum1;
    vsum2 += vsum3;
    vsum0 += vsum2;
    sum0 = vaddvq_f32(vsum0);

    for (; w < width; ++w) {
      sum0 += m_ptr0[0] * v_ptr0[0];
      m_ptr0++;
      v_ptr0++;
    }
    *out_ptr0++ = sum0;
  }
}
#endif

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_FP16_GEMV_H_
