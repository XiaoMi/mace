// Copyright 2019 The MACE Authors. All Rights Reserved.
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


#include "mace/ops/arm/fp32/gemv.h"

#include <arm_neon.h>
#include <algorithm>

#include "mace/utils/math.h"

#if !defined(__aarch64__)
float vaddvq_f32(float32x4_t v) {
  float32x2_t _sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
  _sum = vpadd_f32(_sum, _sum);
  return vget_lane_f32(_sum, 0);
}
#endif

// Disable unroll by default, since cache set conflict could be significant
// #define MACE_GEMV_UNROLL 1

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

MaceStatus Gemv::Compute(const OpContext *context,
                         const Tensor *lhs,
                         const Tensor *rhs,
                         const Tensor *bias,
                         const index_t batch,
                         const index_t lhs_height,
                         const index_t lhs_width,
                         const bool lhs_batched,
                         const bool rhs_batched,
                         Tensor *output) {
  MACE_UNUSED(context);

  MACE_CHECK(output->size() == batch * lhs_height,
             "Need resize output tensor before call gemv.");

  Tensor::MappingGuard lhs_guard(lhs);
  Tensor::MappingGuard rhs_guard(rhs);
  Tensor::MappingGuard bias_guard(bias);
  Tensor::MappingGuard output_guard(output);

  const float *lhs_data = lhs->data<float>();
  const float *rhs_data = rhs->data<float>();
  const float *bias_data = nullptr;
  if (bias) {
    bias_data = bias->data<float>();
  }
  float *output_data = output->mutable_data<float>();

  const index_t h_block_size = 4;
  const index_t h_block_count = RoundUpDiv(lhs_height, h_block_size);
  const index_t w_block_size = 8;
  const index_t w_block_count = lhs_width / w_block_size;
  const index_t w_remain = lhs_width - w_block_size * w_block_count;

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t h_block_idx = start1; h_block_idx < end1;
           h_block_idx += step1) {
        const index_t h_start = h_block_idx * h_block_size;
        const float
            *lhs_ptr = lhs_data
            + static_cast<index_t>(lhs_batched) * b * lhs_height * lhs_width
            + lhs_width * h_start;
        const float *rhs_ptr =
            rhs_data + static_cast<index_t>(rhs_batched) * b * lhs_width;
        float
            *ret_ptr = output_data + b * lhs_height + h_start;

        const index_t h_block_len =
            std::min(h_block_size, lhs_height - h_start);

#ifdef MACE_GEMV_UNROLL
        if (h_block_len == 4) {
        float32x4_t vo0 = vdupq_n_f32(0);
        float32x4_t vo1 = vdupq_n_f32(0);
        float32x4_t vo2 = vdupq_n_f32(0);
        float32x4_t vo3 = vdupq_n_f32(0);

        index_t r_w_block_count = w_block_count;
        // just make compiler happy
        MACE_UNUSED(r_w_block_count);

        // Register layout: (4x8) x (8,1)
        //
        //                                      +----+
        //                                      |d16 |
        //                                      | .  |
        //                              Rhs     +----+
        //                                      |d17 |
        //                                      | .  |
        //                                      +----+
        //                                      |d18 |
        //                                      | .  |
        //                                      +----+
        //                                      |d19 |
        //                                      | .  |
        //                                      +----+
        //
        //                                      |    |
        //
        //                      Lhs             |    |
        //
        //  +------+------+----+-----+ - - - -  +----+
        //  | d0 . | d1 .| d2 .| d3 .|          |vo0 |
        //  | d4 . | d5 .| d6 .| d7 .|          |vo1 |
        //  | d8 . | d9 .| d10.| d11.|          |vo2 |
        //  | d12. | d13.| d14.| d15.|          |vo3 |
        //  +------+-----+-----+-----+ - - - -  +----+
        //
        //                                    Accumulator
        //

#if not defined(__aarch64__)
        asm volatile(
        "cmp %[r_w_block_count], #0\n"
        "beq 0f\n"

        "lsl r5, %[lhs_width], #2\n"

        "mov r0, %[rhs_ptr]\n"
        "mov r1, %[lhs_ptr]\n"
        "add r2, r1, r5\n"
        "add r3, r2, r5\n"
        "add r4, r3, r5\n"

        // prelogue
        "vld1.f32 {d16-d17}, [r0]!\n"
        "vld1.f32 {d18-d19}, [r0]!\n"

        "vld1.f32 {d0-d1}, [r1]!\n"
        "vld1.f32 {d2-d3}, [r1]!\n"
        "vld1.f32 {d4-d5}, [r2]!\n"
        "vld1.f32 {d6-d7}, [r2]!\n"
        "vld1.f32 {d8-d9}, [r3]!\n"
        "vld1.f32 {d10-d11}, [r3]!\n"
        "vld1.f32 {d12-d13}, [r4]!\n"
        "vld1.f32 {d14-d15}, [r4]!\n"

        "subs %[r_w_block_count], #1\n"
        "beq 1f\n"

        "2: \n"
        "vmla.f32 %q[vo0], q0, q8\n"
        "vmla.f32 %q[vo1], q2, q8\n"
        "vmla.f32 %q[vo2], q4, q8\n"
        "vmla.f32 %q[vo3], q6, q8\n"

        "vld1.f32 {d0-d1}, [r1]!\n"
        "vld1.f32 {d4-d5}, [r2]!\n"
        "vld1.f32 {d8-d9}, [r3]!\n"
        "vld1.f32 {d12-d13}, [r4]!\n"
        "vld1.f32 {d16-d17}, [r0]!\n"

        "vmla.f32 %q[vo0], q1, q9\n"
        "vmla.f32 %q[vo1], q3, q9\n"
        "vmla.f32 %q[vo2], q5, q9\n"
        "vmla.f32 %q[vo3], q7, q9\n"

        "subs %[r_w_block_count], #1\n"

        "vld1.f32 {d2-d3}, [r1]!\n"
        "vld1.f32 {d6-d7}, [r2]!\n"
        "vld1.f32 {d10-d11}, [r3]!\n"
        "vld1.f32 {d14-d15}, [r4]!\n"
        "vld1.f32 {d18-d19}, [r0]!\n"

        "bne 2b\n"

        // prologue
        "1:\n"
        "vmla.f32 %q[vo0], q0, q8\n"
        "vmla.f32 %q[vo1], q2, q8\n"
        "vmla.f32 %q[vo2], q4, q8\n"
        "vmla.f32 %q[vo3], q6, q8\n"

        "vmla.f32 %q[vo0], q1, q9\n"
        "vmla.f32 %q[vo1], q3, q9\n"
        "vmla.f32 %q[vo2], q5, q9\n"
        "vmla.f32 %q[vo3], q7, q9\n"

        "0:\n"
        :  // outputs
        [vo0] "+w"(vo0),
        [vo1] "+w"(vo1),
        [vo2] "+w"(vo2),
        [vo3] "+w"(vo3),
        [r_w_block_count] "+r"(r_w_block_count)
        :  // inputs
        [lhs_ptr] "r"(lhs_ptr), [rhs_ptr] "r"(rhs_ptr),
        [lhs_width] "r"(lhs_width)
        :  // clobbers
        "cc", "memory", "r0", "r1", "r2", "r3", "r4", "r5",
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19");

        lhs_ptr += w_block_count * w_block_size;
        rhs_ptr += w_block_count * w_block_size;
#else
        for (index_t w_block_index = 0; w_block_index < w_block_count;
             ++w_block_index) {
          float32x4_t vr0 = vld1q_f32(rhs_ptr);
          float32x4_t vr0n = vld1q_f32(rhs_ptr + 4);

          float32x4_t vl0 = vld1q_f32(lhs_ptr);
          float32x4_t vl0n = vld1q_f32(lhs_ptr + 4);
          vo0 = vmlaq_f32(vo0, vl0, vr0);
          vo0 = vmlaq_f32(vo0, vl0n, vr0n);

          const float *lhs_ptr1 = lhs_ptr + lhs_width;
          float32x4_t vl1 = vld1q_f32(lhs_ptr1);
          float32x4_t vl1n = vld1q_f32(lhs_ptr1 + 4);
          vo1 = vmlaq_f32(vo1, vl1, vr0);
          vo1 = vmlaq_f32(vo1, vl1n, vr0n);

          const float *lhs_ptr2 = lhs_ptr1 + lhs_width;
          float32x4_t vl2 = vld1q_f32(lhs_ptr2);
          float32x4_t vl2n = vld1q_f32(lhs_ptr2 + 4);
          vo2 = vmlaq_f32(vo2, vl2, vr0);
          vo2 = vmlaq_f32(vo2, vl2n, vr0n);

          const float *lhs_ptr3 = lhs_ptr2 + lhs_width;
          float32x4_t vl3 = vld1q_f32(lhs_ptr3);
          float32x4_t vl3n = vld1q_f32(lhs_ptr3 + 4);
          vo3 = vmlaq_f32(vo3, vl3, vr0);
          vo3 = vmlaq_f32(vo3, vl3n, vr0n);

          lhs_ptr += 8;
          rhs_ptr += 8;
        }
#endif  // __aarch64__
        float32x4_t vo = {
            vaddvq_f32(vo0),
            vaddvq_f32(vo1),
            vaddvq_f32(vo2),
            vaddvq_f32(vo3)
        };
        for (index_t w = 0; w < w_remain; ++w) {
          vo[0] += lhs_ptr[0] * rhs_ptr[0];
          vo[1] += lhs_ptr[lhs_width] * rhs_ptr[0];
          vo[2] += lhs_ptr[lhs_width * 2] * rhs_ptr[0];
          vo[3] += lhs_ptr[lhs_width * 3] * rhs_ptr[0];
          ++lhs_ptr;
          ++rhs_ptr;
        }

        if (bias) {
          float32x4_t vbias = vdupq_n_f32(0);
          vbias = vld1q_f32(bias_data + h_start);
          vo = vaddq_f32(vo, vbias);
        }

        vst1q_f32(ret_ptr, vo);
      } else {  // h_block_len < 4
#endif  // MACE_GEMV_UNROLL
        const float *tmp_lhs_ptr = lhs_ptr;
        const float *tmp_rhs_ptr = rhs_ptr;
        for (index_t h = 0; h < h_block_len; ++h) {
          lhs_ptr = tmp_lhs_ptr + h * lhs_width;
          rhs_ptr = tmp_rhs_ptr;

          float s0 = bias ? bias_data[h_start + h] : 0;

          if (w_block_count) {
#if not defined(__aarch64__)
            index_t r_w_block_count = w_block_count;
            float32x4_t vo = vdupq_n_f32(0.f);

            asm volatile(
            "mov r0, #0\n"
            "vdup.f32 q2, r0\n"
            "vdup.f32 q3, r0\n"

            // prelogue
            "vld1.f32 {d16-d17}, [%[rhs_ptr]]!\n"
            "vld1.f32 {d18-d19}, [%[rhs_ptr]]!\n"

            "subs %[r_w_block_count], #1\n"

            "vld1.f32 {d0-d1}, [%[lhs_ptr]]!\n"
            "vld1.f32 {d2-d3}, [%[lhs_ptr]]!\n"

            "beq 1f\n"

            "0: \n"
            "vmla.f32 q2, q0, q8\n"
            "vld1.f32 {d0-d1}, [%[lhs_ptr]]!\n"
            "vld1.f32 {d16-d17}, [%[rhs_ptr]]!\n"

            "subs %[r_w_block_count], #1\n"

            "vmla.f32 q3, q1, q9\n"
            "vld1.f32 {d2-d3}, [%[lhs_ptr]]!\n"
            "vld1.f32 {d18-d19}, [%[rhs_ptr]]!\n"

            "bne 0b\n"

            // prologue
            "1:\n"
            "vmla.f32 q2, q0, q8\n"
            "vmla.f32 q3, q1, q9\n"
            "vaddq.f32 %q[vo], q2, q3\n"
            :  // outputs
            [r_w_block_count] "+r"(r_w_block_count),
            [lhs_ptr] "+r"(lhs_ptr),
            [rhs_ptr] "+r"(rhs_ptr),
            [vo] "+w"(vo)
            :  // inputs
            :  // clobbers
            "cc", "memory", "r0",
            "d0", "d1", "d2", "d3",  // lhs
            "d4", "d5", "d6", "d7", // output
            "d16", "d17", "d18", "d19"  // rhs
            );

            s0 += vaddvq_f32(vo);
#else
            float32x4_t vo0 = vdupq_n_f32(0);
            float32x4_t vo0n = vdupq_n_f32(0);
            for (index_t w = 0; w < w_block_count; ++w) {
              float32x4_t vr0 = vld1q_f32(rhs_ptr);
              float32x4_t vr0n = vld1q_f32(rhs_ptr + 4);

              float32x4_t vl0 = vld1q_f32(lhs_ptr);
              float32x4_t vl0n = vld1q_f32(lhs_ptr + 4);

              vo0 = vmlaq_f32(vo0, vl0, vr0);
              vo0n = vmlaq_f32(vo0n, vl0n, vr0n);

              lhs_ptr += 8;
              rhs_ptr += 8;
            }  // w
            vo0 = vaddq_f32(vo0, vo0n);
            s0 += vaddvq_f32(vo0);
#endif  // __aarch64__
          }  // if
          for (index_t w = 0; w < w_remain; ++w) {
            s0 += lhs_ptr[0] * rhs_ptr[0];
            ++lhs_ptr;
            ++rhs_ptr;
          }  // w

          ret_ptr[h] = s0;
        }  // h

#ifdef MACE_GEMV_UNROLL
        }  // if
#endif  // MACE_GEMV_UNROLL
      }  // h_block_idx
    }  // b
  }, 0, batch, 1, 0, h_block_count, 1);

  return MaceStatus::MACE_SUCCESS;
}

#if defined(vaddvq_f32)
#undef vaddvq_f32
#endif

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace
