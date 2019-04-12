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

#include "mace/ops/arm/q8/eltwise.h"

#include <arm_neon.h>
#include <algorithm>

#include "mace/ops/common/gemmlowp_util.h"
#include "mace/utils/logging.h"

namespace mace {
namespace ops {
namespace arm {
namespace q8 {

MaceStatus Eltwise::Compute(const OpContext *context,
                            const Tensor *input0,
                            const Tensor *input1,
                            Tensor *output) {
  MACE_UNUSED(context);
  MACE_CHECK(type_ == SUM || type_ == SUB,
             "Quantized Elementwise only support SUM and SUB now.");

  constexpr int left_shift = 20;
  const double doubled_scale = 2 * std::max(input0->scale(), input1->scale());
  const double adjusted_input0_scale = input0->scale() / doubled_scale;
  const double adjusted_input1_scale = input1->scale() / doubled_scale;
  const double adjusted_output_scale =
      doubled_scale / ((1 << left_shift) * output->scale());

  int32_t input0_multiplier;
  int32_t input1_multiplier;
  int32_t output_multiplier;
  int32_t input0_shift;
  int32_t input1_shift;
  int32_t output_shift;
  QuantizeMultiplier(adjusted_input0_scale, &input0_multiplier, &input0_shift);
  QuantizeMultiplier(adjusted_input1_scale, &input1_multiplier, &input1_shift);
  QuantizeMultiplier(adjusted_output_scale, &output_multiplier, &output_shift);

  Tensor::MappingGuard input0_guard(input0);
  Tensor::MappingGuard input1_guard(input1);
  Tensor::MappingGuard output_guard(output);

  auto input0_ptr = input0->data<uint8_t>();
  auto input1_ptr = input1->data<uint8_t>();
  auto output_ptr = output->mutable_data<uint8_t>();

  utils::ThreadPool &thread_pool =
      context->device()->cpu_runtime()->thread_pool();
  thread_pool.Compute1D(
      [=](index_t start, index_t end, index_t step) {
        for (index_t i = start; i < end; i += step) {
          const auto input0_val = vld1_u8(input0_ptr + i);
          const auto input1_val = vld1_u8(input1_ptr + i);
          const auto input0_val_s16 =
              vreinterpretq_s16_u16(vmovl_u8(input0_val));
          const auto input1_val_s16 =
              vreinterpretq_s16_u16(vmovl_u8(input1_val));
          const auto offset_input0 =
              vaddq_s16(input0_val_s16, vdupq_n_s16(-input0->zero_point()));
          const auto offset_input1 =
              vaddq_s16(input1_val_s16, vdupq_n_s16(-input1->zero_point()));
          auto input0_low_s32 = vmovl_s16(vget_low_s16(offset_input0));
          auto input0_high_s32 = vmovl_s16(vget_high_s16(offset_input0));
          auto input1_low_s32 = vmovl_s16(vget_low_s16(offset_input1));
          auto input1_high_s32 = vmovl_s16(vget_high_s16(offset_input1));
          const auto left_shift_dup = vdupq_n_s32(left_shift);
          input0_low_s32 = vshlq_s32(input0_low_s32, left_shift_dup);
          input0_high_s32 = vshlq_s32(input0_high_s32, left_shift_dup);
          input1_low_s32 = vshlq_s32(input1_low_s32, left_shift_dup);
          input1_high_s32 = vshlq_s32(input1_high_s32, left_shift_dup);
          input0_low_s32 = vqrdmulhq_n_s32(input0_low_s32, input0_multiplier);
          input0_high_s32 = vqrdmulhq_n_s32(input0_high_s32, input0_multiplier);
          input1_low_s32 = vqrdmulhq_n_s32(input1_low_s32, input1_multiplier);
          input1_high_s32 = vqrdmulhq_n_s32(input1_high_s32, input1_multiplier);
          const auto input0_shift_dup = vdupq_n_s32(input0_shift);
          const auto input1_shift_dup = vdupq_n_s32(input1_shift);
          input0_low_s32 = vshlq_s32(input0_low_s32, input0_shift_dup);
          input0_high_s32 = vshlq_s32(input0_high_s32, input0_shift_dup);
          input1_low_s32 = vshlq_s32(input1_low_s32, input1_shift_dup);
          input1_high_s32 = vshlq_s32(input1_high_s32, input1_shift_dup);
          int32x4_t res_low, res_high;
          if (type_ == SUM) {
            res_low = vaddq_s32(input0_low_s32, input1_low_s32);
            res_high = vaddq_s32(input0_high_s32, input1_high_s32);
          } else {
            res_low = vsubq_s32(input0_low_s32, input1_low_s32);
            res_high = vsubq_s32(input0_high_s32, input1_high_s32);
          }
          res_low = vqrdmulhq_n_s32(res_low, output_multiplier);
          res_high = vqrdmulhq_n_s32(res_high, output_multiplier);
          res_low = gemmlowp::RoundingDivideByPOT(res_low, -output_shift);
          res_high = gemmlowp::RoundingDivideByPOT(res_high, -output_shift);
          const auto res_low_s16 = vmovn_s32(res_low);
          const auto res_high_s16 = vmovn_s32(res_high);
          const auto output_val =
              vaddq_s16(vcombine_s16(res_low_s16, res_high_s16),
                        vdupq_n_s16(output->zero_point()));
          vst1_u8(output_ptr + i, vqmovun_s16(output_val));
        }
      },
      0, output->size() - 7, 8);

  index_t handled_output_size = output->size() - output->size() % 8;

  thread_pool.Compute1D(
      [=](index_t start, index_t end, index_t step) {
        for (index_t i = start; i < end; i += step) {
          const int32_t offset_input0 = input0_ptr[i] - input0->zero_point();
          const int32_t offset_input1 = input1_ptr[i] - input1->zero_point();
          const int32_t shifted_input0 = offset_input0 * (1 << left_shift);
          const int32_t shifted_input1 = offset_input1 * (1 << left_shift);
          const int32_t multiplied_input0 = gemmlowp::RoundingDivideByPOT(
              gemmlowp::SaturatingRoundingDoublingHighMul(shifted_input0,
                                                          input0_multiplier),
              -input0_shift);
          const int32_t multiplied_input1 = gemmlowp::RoundingDivideByPOT(
              gemmlowp::SaturatingRoundingDoublingHighMul(shifted_input1,
                                                          input1_multiplier),
              -input1_shift);

          int32_t res;
          if (type_ == SUM) {
            res = multiplied_input0 + multiplied_input1;
          } else {
            res = multiplied_input0 - multiplied_input1;
          }

          const int32_t output_val =
              gemmlowp::RoundingDivideByPOT(
                  gemmlowp::SaturatingRoundingDoublingHighMul(
                      res, output_multiplier),
                  -output_shift) +
              output->zero_point();
          output_ptr[i] = Saturate<uint8_t>(output_val);
        }
      },
      handled_output_size, output->size(), 1);

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace q8
}  // namespace arm
}  // namespace ops
}  // namespace mace
