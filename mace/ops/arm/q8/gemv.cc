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


#include "mace/ops/arm/q8/gemv.h"

#include <arm_neon.h>
#include <algorithm>

#include "mace/utils/utils.h"
#include "mace/utils/quantize.h"

#if !defined(__aarch64__)

#define vmlal_high_s16(c, a, b) vmlal_s16(c, vget_high_s16(a), vget_high_s16(b))

#define vaddvq_s32(v) ((v)[0] + (v)[1] + (v)[2] + (v)[3])

#endif

namespace mace {
namespace ops {
namespace arm {
namespace q8 {

template<typename OUTPUT_TYPE>
MaceStatus Gemv<OUTPUT_TYPE>::Compute(const OpContext *context,
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

  bool is_output_type_uint8 =
      DataTypeToEnum<OUTPUT_TYPE>::value == DataType::DT_UINT8;
  Tensor::MappingGuard lhs_guard(lhs);
  Tensor::MappingGuard rhs_guard(rhs);
  Tensor::MappingGuard bias_guard(bias);
  Tensor::MappingGuard output_guard(output);

  float output_multiplier_float = 0.0;
  int32_t output_multiplier = 0;
  int32_t output_shift = 0;
  if (is_output_type_uint8) {
    MACE_CHECK(output->scale() > 0, "output scale must not be zero");
    output_multiplier_float = lhs->scale() * rhs->scale() / output->scale();
    GetOutputMultiplierAndShift(lhs->scale(),
                                rhs->scale(),
                                output->scale(),
                                &output_multiplier,
                                &output_shift);
  }
  const index_t h_block_size = 4;
  const index_t h_block_count = RoundUpDiv(lhs_height, h_block_size);

#pragma omp parallel for collapse(2) schedule(runtime)
  for (index_t b = 0; b < batch; ++b) {
    for (index_t h_block_idx = 0; h_block_idx < h_block_count; ++h_block_idx) {
      // TODO(liyin): it can be put it outside the loop,
      // but openmp limits param count
      const index_t w_block_size = 16;
      const index_t w_block_count = lhs_width / w_block_size;
      const index_t w_remain = lhs_width - w_block_size * w_block_count;

      uint8_t lhs_zero_point = static_cast<uint8_t>(lhs->zero_point());
      uint8_t rhs_zero_point = static_cast<uint8_t>(rhs->zero_point());

      const uint8_t *lhs_data = lhs->data<uint8_t>();
      const uint8_t *rhs_data = rhs->data<uint8_t>();
      const int32_t *bias_data = nullptr;
      if (bias) {
        bias_data = bias->data<int32_t>();
      }
      OUTPUT_TYPE *output_data = output->mutable_data<OUTPUT_TYPE>();

      int32x4_t voutput_multiplier = vdupq_n_s32(output_multiplier);
      int32x4_t voutput_shift_left = vdupq_n_s32(-output_shift);

      uint8x8_t
          vlhs_zero_point = vdup_n_u8(lhs_zero_point);
      uint8x8_t
          vrhs_zero_point = vdup_n_u8(rhs_zero_point);

      const uint8_t
          *lhs_ptr = lhs_data
          + static_cast<index_t>(lhs_batched) * b * lhs_height * lhs_width
          + lhs_width * h_block_idx * h_block_size;
      const uint8_t *rhs_ptr =
          rhs_data + static_cast<index_t>(rhs_batched) * b * lhs_width;
      OUTPUT_TYPE
          *ret_ptr = output_data + b * lhs_height + h_block_idx * h_block_size;

      const index_t h_block_len =
          std::min(h_block_size, lhs_height - h_block_idx * h_block_size);
      const index_t h_offset = h_block_idx * h_block_size;

      if (h_block_len == 4) {
        int32x4_t vo0 = vdupq_n_s32(0);
        int32x4_t vo1 = vdupq_n_s32(0);
        int32x4_t vo2 = vdupq_n_s32(0);
        int32x4_t vo3 = vdupq_n_s32(0);

        index_t r_w_block_count = w_block_count;
        // just make compiler happy
        MACE_UNUSED(r_w_block_count);

        // Register layout: (4x16) x (16x1)
        //
        //                                                 +----+
        //                                                 |d16 |
        //                                                 | .  |
        //                                                 | .  |
        //                                                 | .  |
        //                                         Rhs     +----+
        //                                                 |d17 |
        //                                                 | .  |
        //                                                 | .  |
        //                                                 | .  |
        //                                                 +----+
        //                                                 |d18 |
        //                                                 | .  |
        //                                                 | .  |
        //                                                 | .  |
        //                                                 +----+
        //                                                 |d19 |
        //                                                 | .  |
        //                                                 | .  |
        //                                                 | .  |
        //                                                 +----+
        //
        //                                                 |    |
        //
        //                      Lhs                        |    |
        //
        //  +--------+--------+--------+--------+ - - - -  +----+
        //  | d0 ... | d1 ... | d2 ... | d3 ... |          |vo0 |
        //  | d4 ... | d5 ... | d6 ... | d7 ... |          |vo1 |
        //  | d8 ... | d9 ... | d10... | d11... |          |vo2 |
        //  | d12... | d13... | d14... | d15... |          |vo3 |
        //  +--------+--------+--------+--------+ - - - -  +----+
        //
        //                                               Accumulator
        //

#if not defined(__aarch64__)
        asm volatile(
        "cmp %[r_w_block_count], #0\n"
        "beq 0f\n"

        "mov r0, %[rhs_ptr]\n"
        "mov r1, %[lhs_ptr]\n"
        "add r2, r1, %[lhs_width]\n"
        "add r3, r2, %[lhs_width]\n"
        "add r4, r3, %[lhs_width]\n"

        "vdup.u8 d20, %[rhs_zero_point]\n"
        "vdup.u8 d21, %[lhs_zero_point]\n"

        // prelogue
        "vld1.8 d16, [r0]!\n"
        "vld1.8 d18, [r0]!\n"

        "vld1.8 d0, [r1]!\n"
        "vld1.8 d2, [r1]!\n"
        "vld1.8 d4, [r2]!\n"
        "vld1.8 d6, [r2]!\n"
        "vld1.8 d8, [r3]!\n"
        "vld1.8 d10, [r3]!\n"
        "vld1.8 d12, [r4]!\n"
        "vld1.8 d14, [r4]!\n"

        "subs %[r_w_block_count], #1\n"
        "beq 1f\n"

        "2: \n"
        "vsubl.u8 q8, d16, d20\n"
        "vsubl.u8 q9, d18, d20\n"

        "vsubl.u8 q0, d0, d21\n"
        "vsubl.u8 q1, d2, d21\n"
        "vsubl.u8 q2, d4, d21\n"
        "vsubl.u8 q3, d6, d21\n"
        "vsubl.u8 q4, d8, d21\n"
        "vsubl.u8 q5, d10, d21\n"
        "vsubl.u8 q6, d12, d21\n"
        "vsubl.u8 q7, d14, d21\n"

        "vmlal.s16 %q[vo0], d0, d16\n"
        "vmlal.s16 %q[vo1], d4, d16\n"
        "vmlal.s16 %q[vo2], d8, d16\n"
        "vmlal.s16 %q[vo3], d12, d16\n"

        "vld1.8 d0, [r1]!\n"
        "vld1.8 d4, [r2]!\n"
        "vld1.8 d8, [r3]!\n"
        "vld1.8 d12, [r4]!\n"
        "vld1.8 d16, [r0]!\n"

        "vmlal.s16 %q[vo0], d2, d18\n"
        "vmlal.s16 %q[vo1], d6, d18\n"
        "vmlal.s16 %q[vo2], d10, d18\n"
        "vmlal.s16 %q[vo3], d14, d18\n"

        "vld1.8 d2, [r1]!\n"
        "vld1.8 d6, [r2]!\n"
        "vld1.8 d10, [r3]!\n"
        "vld1.8 d14, [r4]!\n"
        "vld1.8 d18, [r0]!\n"

        "vmlal.s16 %q[vo0], d1, d17\n"
        "vmlal.s16 %q[vo1], d5, d17\n"
        "vmlal.s16 %q[vo2], d9, d17\n"
        "vmlal.s16 %q[vo3], d13, d17\n"

        "subs %[r_w_block_count], #1\n"
        "vmlal.s16 %q[vo0], d3, d19\n"
        "vmlal.s16 %q[vo1], d7, d19\n"
        "vmlal.s16 %q[vo2], d11, d19\n"
        "vmlal.s16 %q[vo3], d15, d19\n"

        "bne 2b\n"

        // prologue
        "1:\n"
        "vsubl.u8 q8, d16, d20\n"
        "vsubl.u8 q9, d18, d20\n"

        "vsubl.u8 q0, d0, d21\n"
        "vsubl.u8 q1, d2, d21\n"
        "vsubl.u8 q2, d4, d21\n"
        "vsubl.u8 q3, d6, d21\n"
        "vsubl.u8 q4, d8, d21\n"
        "vsubl.u8 q5, d10, d21\n"
        "vsubl.u8 q6, d12, d21\n"
        "vsubl.u8 q7, d14, d21\n"

        "vmlal.s16 %q[vo0], d0, d16\n"
        "vmlal.s16 %q[vo1], d4, d16\n"
        "vmlal.s16 %q[vo2], d8, d16\n"
        "vmlal.s16 %q[vo3], d12, d16\n"

        "vmlal.s16 %q[vo0], d1, d17\n"
        "vmlal.s16 %q[vo1], d5, d17\n"
        "vmlal.s16 %q[vo2], d9, d17\n"
        "vmlal.s16 %q[vo3], d13, d17\n"

        "vmlal.s16 %q[vo0], d2, d18\n"
        "vmlal.s16 %q[vo1], d6, d18\n"
        "vmlal.s16 %q[vo2], d10, d18\n"
        "vmlal.s16 %q[vo3], d14, d18\n"

        "vmlal.s16 %q[vo0], d3, d19\n"
        "vmlal.s16 %q[vo1], d7, d19\n"
        "vmlal.s16 %q[vo2], d11, d19\n"
        "vmlal.s16 %q[vo3], d15, d19\n"

        "0:\n"
        :  // outputs
        [vo0] "+w"(vo0),
        [vo1] "+w"(vo1),
        [vo2] "+w"(vo2),
        [vo3] "+w"(vo3),
        [r_w_block_count] "+r"(r_w_block_count)
        :  // inputs
        [lhs_ptr] "r"(lhs_ptr), [rhs_ptr] "r"(rhs_ptr),
        [lhs_width] "r"(lhs_width),
        [lhs_zero_point] "r"(lhs_zero_point),
        [rhs_zero_point] "r"(rhs_zero_point)
        :  // clobbers
        "cc", "memory", "r0", "r1", "r2", "r3", "r4",
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21");

        lhs_ptr += w_block_count * w_block_size;
        rhs_ptr += w_block_count * w_block_size;
#else
        for (index_t w_block_index = 0; w_block_index < w_block_count;
             ++w_block_index) {
          uint8x8_t vr0 = vld1_u8(rhs_ptr);
          int16x8_t
              vxr0 = vreinterpretq_s16_u16(vsubl_u8(vr0, vrhs_zero_point));
          uint8x8_t vr0n = vld1_u8(rhs_ptr + 8);
          int16x8_t
              vxr0n = vreinterpretq_s16_u16(vsubl_u8(vr0n, vrhs_zero_point));

          uint8x8_t vl0 = vld1_u8(lhs_ptr);
          int16x8_t
              vxl0 = vreinterpretq_s16_u16(vsubl_u8(vl0, vlhs_zero_point));
          uint8x8_t vl0n = vld1_u8(lhs_ptr + 8);
          int16x8_t
              vxl0n = vreinterpretq_s16_u16(vsubl_u8(vl0n, vlhs_zero_point));

          vo0 = vmlal_s16(vo0, vget_low_s16(vxl0), vget_low_s16(vxr0));
          vo0 = vmlal_high_s16(vo0, vxl0, vxr0);
          vo0 = vmlal_s16(vo0, vget_low_s16(vxl0n), vget_low_s16(vxr0n));
          vo0 = vmlal_high_s16(vo0, vxl0n, vxr0n);

          const uint8_t *lhs_ptr1 = lhs_ptr + lhs_width;

          uint8x8_t vl1 = vld1_u8(lhs_ptr1);
          int16x8_t
              vxl1 = vreinterpretq_s16_u16(vsubl_u8(vl1, vlhs_zero_point));
          uint8x8_t vl1n = vld1_u8(lhs_ptr1 + 8);
          int16x8_t
              vxl1n = vreinterpretq_s16_u16(vsubl_u8(vl1n, vlhs_zero_point));

          vo1 = vmlal_s16(vo1, vget_low_s16(vxl1), vget_low_s16(vxr0));
          vo1 = vmlal_high_s16(vo1, vxl1, vxr0);
          vo1 = vmlal_s16(vo1, vget_low_s16(vxl1n), vget_low_s16(vxr0n));
          vo1 = vmlal_high_s16(vo1, vxl1n, vxr0n);

          const uint8_t *lhs_ptr2 = lhs_ptr1 + lhs_width;

          uint8x8_t vl2 = vld1_u8(lhs_ptr2);
          int16x8_t
              vxl2 = vreinterpretq_s16_u16(vsubl_u8(vl2, vlhs_zero_point));
          uint8x8_t vl2n = vld1_u8(lhs_ptr2 + 8);
          int16x8_t
              vxl2n = vreinterpretq_s16_u16(vsubl_u8(vl2n, vlhs_zero_point));

          vo2 = vmlal_s16(vo2, vget_low_s16(vxl2), vget_low_s16(vxr0));
          vo2 = vmlal_high_s16(vo2, vxl2, vxr0);
          vo2 = vmlal_s16(vo2, vget_low_s16(vxl2n), vget_low_s16(vxr0n));
          vo2 = vmlal_high_s16(vo2, vxl2n, vxr0n);

          const uint8_t *lhs_ptr3 = lhs_ptr2 + lhs_width;

          uint8x8_t vl3 = vld1_u8(lhs_ptr3);
          int16x8_t
              vxl3 = vreinterpretq_s16_u16(vsubl_u8(vl3, vlhs_zero_point));
          uint8x8_t vl3n = vld1_u8(lhs_ptr3 + 8);
          int16x8_t
              vxl3n = vreinterpretq_s16_u16(vsubl_u8(vl3n, vlhs_zero_point));

          vo3 = vmlal_s16(vo3, vget_low_s16(vxl3), vget_low_s16(vxr0));
          vo3 = vmlal_high_s16(vo3, vxl3, vxr0);
          vo3 = vmlal_s16(vo3, vget_low_s16(vxl3n), vget_low_s16(vxr0n));
          vo3 = vmlal_high_s16(vo3, vxl3n, vxr0n);

          lhs_ptr += 16;
          rhs_ptr += 16;
        }
#endif  // __aarch64__
        int32x4_t vo = {vaddvq_s32(vo0),
                        vaddvq_s32(vo1),
                        vaddvq_s32(vo2),
                        vaddvq_s32(vo3)};

        for (index_t w = 0; w < w_remain; ++w) {
          vo[0] +=
              (lhs_ptr[0] - lhs_zero_point) * (rhs_ptr[0] - rhs_zero_point);
          vo[1] += (lhs_ptr[lhs_width] - lhs_zero_point)
              * (rhs_ptr[0] - rhs_zero_point);
          vo[2] += (lhs_ptr[lhs_width * 2] - lhs_zero_point)
              * (rhs_ptr[0] - rhs_zero_point);
          vo[3] += (lhs_ptr[lhs_width * 3] - lhs_zero_point)
              * (rhs_ptr[0] - rhs_zero_point);
          ++lhs_ptr;
          ++rhs_ptr;
        }

        int32x4_t vbias = vdupq_n_s32(0);
        if (bias) {
          vbias = vld1q_s32(bias_data + h_offset);
        }
        vo = vaddq_s32(vo, vbias);

        if (is_output_type_uint8) {
          int32x4_t vo_mul = vqrdmulhq_s32(vo, voutput_multiplier);
          int32x4_t
              fixup = vshrq_n_s32(vandq_s32(vo_mul, voutput_shift_left), 31);
          int32x4_t fixed_up_x = vqaddq_s32(vo_mul, fixup);
          int32x4_t
              vo_rescale_int32 = vrshlq_s32(fixed_up_x, voutput_shift_left);

          int16x4_t vo_rescale_int16 = vqmovn_s32(vo_rescale_int32);
          uint8x8_t vo_rescale_uint8 =
              vqmovun_s16(vcombine_s16(vo_rescale_int16, vo_rescale_int16));

          ret_ptr[0] = vo_rescale_uint8[0];
          ret_ptr[1] = vo_rescale_uint8[1];
          ret_ptr[2] = vo_rescale_uint8[2];
          ret_ptr[3] = vo_rescale_uint8[3];
        } else {
          ret_ptr[0] = vo[0];
          ret_ptr[1] = vo[1];
          ret_ptr[2] = vo[2];
          ret_ptr[3] = vo[3];
        }
      } else {  // h_block_len < 4
        // TODO(liyin): handle here case by case (1,2,3) to accelerate
        const uint8_t *tmp_lhs_ptr = lhs_ptr;
        const uint8_t *tmp_rhs_ptr = rhs_ptr;
        for (index_t h = 0; h < h_block_len; ++h) {
          lhs_ptr = tmp_lhs_ptr + h * lhs_width;
          rhs_ptr = tmp_rhs_ptr;
          int32x4_t vo0 = vdupq_n_s32(0);
          for (index_t w = 0; w < w_block_count; ++w) {
            uint8x8_t vr0 = vld1_u8(rhs_ptr);
            int16x8_t
                vxr0 = vreinterpretq_s16_u16(vsubl_u8(vr0, vrhs_zero_point));
            uint8x8_t vr0n = vld1_u8(rhs_ptr + 8);
            int16x8_t
                vxr0n = vreinterpretq_s16_u16(vsubl_u8(vr0n, vrhs_zero_point));

            uint8x8_t vl0 = vld1_u8(lhs_ptr);
            int16x8_t
                vxl0 = vreinterpretq_s16_u16(vsubl_u8(vl0, vlhs_zero_point));
            uint8x8_t vl0n = vld1_u8(lhs_ptr + 8);
            int16x8_t
                vxl0n = vreinterpretq_s16_u16(vsubl_u8(vl0n, vlhs_zero_point));

            vo0 = vmlal_s16(vo0, vget_low_s16(vxl0), vget_low_s16(vxr0));
            vo0 = vmlal_high_s16(vo0, vxl0, vxr0);
            vo0 = vmlal_s16(vo0, vget_low_s16(vxl0n), vget_low_s16(vxr0n));
            vo0 = vmlal_high_s16(vo0, vxl0n, vxr0n);

            lhs_ptr += 16;
            rhs_ptr += 16;
          }  // w
          int32_t s0 = vaddvq_s32(vo0) + (bias ? bias_data[h_offset + h] : 0);
          for (index_t w = 0; w < w_remain; ++w) {
            s0 += (lhs_ptr[0] - lhs_zero_point) * (rhs_ptr[0] - rhs_zero_point);
            ++lhs_ptr;
            ++rhs_ptr;
          }  // w

          if (is_output_type_uint8) {
            ret_ptr[h] =
                Saturate<uint8_t>(std::roundf(s0 * output_multiplier_float));
          } else {
            ret_ptr[h] = s0;
          }
        }  // h
      }  // if
    }  // h_block_idx
  }  // b

  return MaceStatus::MACE_SUCCESS;
}

template
class Gemv<uint8_t>;
template
class Gemv<int32_t>;

}  // namespace q8
}  // namespace arm
}  // namespace ops
}  // namespace mace

#if defined(vmlal_high_s16)
#undef vmlal_high_s16
#undef vaddvq_s32
#endif
