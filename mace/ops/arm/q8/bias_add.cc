// Copyright 2020 The MACE Authors. All Rights sumerved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expsums or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mace/ops/arm/base/bias_add.h"

#include <arm_neon.h>

#include "mace/core/quantize.h"

namespace mace {
namespace ops {
namespace arm {

template <>
template <int Dim>
void BiasAdd<uint8_t>::AddBiasNCHW(utils::ThreadPool *thread_pool,
                                   const Tensor *input,
                                   const Tensor *bias,
                                   Tensor *output) {
  auto input_data = input->data<uint8_t>();
  auto bias_data = bias->data<uint8_t>();
  auto output_data = output->mutable_data<uint8_t>();

  const index_t batch = input->dim(0);
  const index_t channels = input->dim(1);
  const index_t image_size = input->dim(2) * input->dim(3);
  const index_t block_count = image_size / 8;
  const index_t remain = image_size % 8;

  constexpr int left_shift = 20;
  const double doubled_scale = 2 * std::max(input->scale(), bias->scale());
  const double adjusted_input_scale = input->scale() / doubled_scale;
  const double adjusted_bias_scale = bias->scale() / doubled_scale;
  const double adjusted_output_scale =
      doubled_scale / ((1 << left_shift) * output->scale());

  int32_t input_multiplier;
  int32_t bias_multiplier;
  int32_t output_multiplier;
  int32_t input_shift;
  int32_t bias_shift;
  int32_t output_shift;
  QuantizeMultiplier(adjusted_input_scale, &input_multiplier, &input_shift);
  QuantizeMultiplier(adjusted_bias_scale, &bias_multiplier, &bias_shift);
  QuantizeMultiplier(adjusted_output_scale, &output_multiplier, &output_shift);
  const auto left_shift_dup = vdupq_n_s32(left_shift);
  const auto input_shift_dup = vdupq_n_s32(input_shift);

  thread_pool->Compute2D(
      [=](index_t start0, index_t end0, index_t step0, index_t start1,
          index_t end1, index_t step1) {
        for (index_t b = start0; b < end0; b += step0) {
          const index_t b_offset = b * channels;
          for (index_t c = start1; c < end1; c += step1) {
            const index_t offset = (b_offset + c) * image_size;
            auto input_ptr = input_data + offset;
            auto output_ptr = output_data + offset;

            const int32_t offset_bias =
                bias_data[bias_index<Dim>(b_offset, c)] - bias->zero_point();
            const int32_t shifted_bias = offset_bias * (1 << left_shift);
            const int32_t multiplied_bias = gemmlowp::RoundingDivideByPOT(
                gemmlowp::SaturatingRoundingDoublingHighMul(shifted_bias,
                                                            bias_multiplier),
                -bias_shift);
            const auto bias_low_s32 = vdupq_n_s32(multiplied_bias);
            const auto bias_high_s32 = vdupq_n_s32(multiplied_bias);

            for (index_t i = 0; i < block_count; ++i) {
              const auto input_val = vld1_u8(input_ptr);
              const auto input_val_s16 =
                  vreinterpretq_s16_u16(vmovl_u8(input_val));
              const auto offset_input =
                  vaddq_s16(input_val_s16, vdupq_n_s16(-input->zero_point()));
              auto input_low_s32 = vmovl_s16(vget_low_s16(offset_input));
              auto input_high_s32 = vmovl_s16(vget_high_s16(offset_input));
              input_low_s32 = vshlq_s32(input_low_s32, left_shift_dup);
              input_high_s32 = vshlq_s32(input_high_s32, left_shift_dup);
              input_low_s32 = vqrdmulhq_n_s32(input_low_s32, input_multiplier);
              input_high_s32 =
                  vqrdmulhq_n_s32(input_high_s32, input_multiplier);
              input_low_s32 = vshlq_s32(input_low_s32, input_shift_dup);
              input_high_s32 = vshlq_s32(input_high_s32, input_shift_dup);
              auto sum_low = vaddq_s32(input_low_s32, bias_low_s32);
              auto sum_high = vaddq_s32(input_high_s32, bias_high_s32);
              sum_low = vqrdmulhq_n_s32(sum_low, output_multiplier);
              sum_high = vqrdmulhq_n_s32(sum_high, output_multiplier);
              sum_low = gemmlowp::RoundingDivideByPOT(sum_low, -output_shift);
              sum_high = gemmlowp::RoundingDivideByPOT(sum_high, -output_shift);
              const auto sum_low_s16 = vmovn_s32(sum_low);
              const auto sum_high_s16 = vmovn_s32(sum_high);
              const auto output_val =
                  vaddq_s16(vcombine_s16(sum_low_s16, sum_high_s16),
                            vdupq_n_s16(output->zero_point()));
              vst1_u8(output_ptr, vqmovun_s16(output_val));

              input_ptr += 8;
              output_ptr += 8;
            }

            for (index_t i = 0; i < remain; ++i) {
              const int32_t offset_input = input_ptr[i] - input->zero_point();
              const int32_t shifted_input = offset_input * (1 << left_shift);
              const int32_t multiplied_input = gemmlowp::RoundingDivideByPOT(
                  gemmlowp::SaturatingRoundingDoublingHighMul(shifted_input,
                                                              input_multiplier),
                  -input_shift);
              int32_t sum = multiplied_input + multiplied_bias;
              const int32_t output_val =
                  gemmlowp::RoundingDivideByPOT(
                      gemmlowp::SaturatingRoundingDoublingHighMul(
                          sum, output_multiplier),
                      -output_shift) +
                  output->zero_point();
              output_ptr[i] = Saturate<uint8_t>(output_val);
            }
          }
        }
      },
      0, batch, 1, 0, channels, 1);
}

template <>
template <int Dim>
void BiasAdd<uint8_t>::AddBiasNHWC(utils::ThreadPool *thread_pool,
                                   const Tensor *input,
                                   const Tensor *bias,
                                   Tensor *output) {
  auto input_data = input->data<uint8_t>();
  auto bias_data = bias->data<uint8_t>();
  auto output_data = output->mutable_data<uint8_t>();

  const std::vector<index_t> &shape = input->shape();
  const index_t channels = *shape.rbegin();
  const index_t block_count = channels / 8;
  const index_t remain = channels % 8;

  constexpr int left_shift = 20;
  const double doubled_scale = 2 * std::max(input->scale(), bias->scale());
  const double adjusted_input_scale = input->scale() / doubled_scale;
  const double adjusted_bias_scale = bias->scale() / doubled_scale;
  const double adjusted_output_scale =
      doubled_scale / ((1 << left_shift) * output->scale());

  int32_t input_multiplier;
  int32_t bias_multiplier;
  int32_t output_multiplier;
  int32_t input_shift;
  int32_t bias_shift;
  int32_t output_shift;
  QuantizeMultiplier(adjusted_input_scale, &input_multiplier, &input_shift);
  QuantizeMultiplier(adjusted_bias_scale, &bias_multiplier, &bias_shift);
  QuantizeMultiplier(adjusted_output_scale, &output_multiplier, &output_shift);
  const auto left_shift_dup = vdupq_n_s32(left_shift);
  const auto input_shift_dup = vdupq_n_s32(input_shift);
  const auto bias_shift_dup = vdupq_n_s32(bias_shift);

  const auto batch = shape[0];
  if (Dim == 2) {
    MACE_CHECK(batch == bias->shape()[0]);
  }
  const index_t fused_hw = std::accumulate(shape.begin() + 1, shape.end() - 1,
                                           1, std::multiplies<index_t>());
  thread_pool->Compute2D(
      [=](index_t start0, index_t end0, index_t step0, index_t start1,
          index_t end1, index_t step1) {
        for (index_t i = start0; i < end0; i += step0) {
          auto offset = i * fused_hw;
          auto bias_offset = i * channels;
          for (index_t j = start1; j < end1; j += step1) {
            index_t pos = (offset + j) * channels;
            auto input_ptr = input_data + pos;
            auto output_ptr = output_data + pos;
            auto bias_ptr = bias_data + bias_index<Dim>(bias_offset, 0);
            for (index_t c = 0; c < block_count; ++c) {
              const auto input_val = vld1_u8(input_ptr);
              const auto bias_val = vld1_u8(bias_ptr);
              const auto input_val_s16 =
                  vreinterpretq_s16_u16(vmovl_u8(input_val));
              const auto bias_val_s16 =
                  vreinterpretq_s16_u16(vmovl_u8(bias_val));
              const auto offset_input =
                  vaddq_s16(input_val_s16, vdupq_n_s16(-input->zero_point()));
              const auto offset_bias =
                  vaddq_s16(bias_val_s16, vdupq_n_s16(-bias->zero_point()));
              auto input_low_s32 = vmovl_s16(vget_low_s16(offset_input));
              auto input_high_s32 = vmovl_s16(vget_high_s16(offset_input));
              auto bias_low_s32 = vmovl_s16(vget_low_s16(offset_bias));
              auto bias_high_s32 = vmovl_s16(vget_high_s16(offset_bias));
              input_low_s32 = vshlq_s32(input_low_s32, left_shift_dup);
              input_high_s32 = vshlq_s32(input_high_s32, left_shift_dup);
              bias_low_s32 = vshlq_s32(bias_low_s32, left_shift_dup);
              bias_high_s32 = vshlq_s32(bias_high_s32, left_shift_dup);
              input_low_s32 = vqrdmulhq_n_s32(input_low_s32, input_multiplier);
              input_high_s32 =
                  vqrdmulhq_n_s32(input_high_s32, input_multiplier);
              bias_low_s32 = vqrdmulhq_n_s32(bias_low_s32, bias_multiplier);
              bias_high_s32 = vqrdmulhq_n_s32(bias_high_s32, bias_multiplier);
              input_low_s32 = vshlq_s32(input_low_s32, input_shift_dup);
              input_high_s32 = vshlq_s32(input_high_s32, input_shift_dup);
              bias_low_s32 = vshlq_s32(bias_low_s32, bias_shift_dup);
              bias_high_s32 = vshlq_s32(bias_high_s32, bias_shift_dup);
              int32x4_t sum_low = vaddq_s32(input_low_s32, bias_low_s32);
              int32x4_t sum_high = vaddq_s32(input_high_s32, bias_high_s32);
              sum_low = vqrdmulhq_n_s32(sum_low, output_multiplier);
              sum_high = vqrdmulhq_n_s32(sum_high, output_multiplier);
              sum_low = gemmlowp::RoundingDivideByPOT(sum_low, -output_shift);
              sum_high = gemmlowp::RoundingDivideByPOT(sum_high, -output_shift);
              const auto sum_low_s16 = vmovn_s32(sum_low);
              const auto sum_high_s16 = vmovn_s32(sum_high);
              const auto output_val =
                  vaddq_s16(vcombine_s16(sum_low_s16, sum_high_s16),
                            vdupq_n_s16(output->zero_point()));
              vst1_u8(output_ptr, vqmovun_s16(output_val));

              input_ptr += 8;
              bias_ptr += 8;
              output_ptr += 8;
            }
            for (index_t c = 0; c < remain; ++c) {
              const int32_t offset_input = input_ptr[c] - input->zero_point();
              const int32_t offset_bias = bias_ptr[c] - bias->zero_point();
              const int32_t shifted_input = offset_input * (1 << left_shift);
              const int32_t shifted_bias = offset_bias * (1 << left_shift);
              const int32_t multiplied_input = gemmlowp::RoundingDivideByPOT(
                  gemmlowp::SaturatingRoundingDoublingHighMul(shifted_input,
                                                              input_multiplier),
                  -input_shift);
              const int32_t multiplied_bias = gemmlowp::RoundingDivideByPOT(
                  gemmlowp::SaturatingRoundingDoublingHighMul(shifted_bias,
                                                              bias_multiplier),
                  -bias_shift);

              int32_t sum = multiplied_input + multiplied_bias;

              const int32_t output_val =
                  gemmlowp::RoundingDivideByPOT(
                      gemmlowp::SaturatingRoundingDoublingHighMul(
                          sum, output_multiplier),
                      -output_shift) +
                  output->zero_point();
              output_ptr[c] = Saturate<uint8_t>(output_val);
            }
          }
        }
      },
      0, batch, 1, 0, fused_hw, 1);
}

template void BiasAdd<uint8_t>::AddBiasNCHW<1>(utils::ThreadPool *thread_pool,
                                               const Tensor *input,
                                               const Tensor *bias,
                                               Tensor *output);
template void BiasAdd<uint8_t>::AddBiasNCHW<2>(utils::ThreadPool *thread_pool,
                                               const Tensor *input,
                                               const Tensor *bias,
                                               Tensor *output);
template void BiasAdd<uint8_t>::AddBiasNHWC<1>(utils::ThreadPool *thread_pool,
                                               const Tensor *input,
                                               const Tensor *bias,
                                               Tensor *output);
template void BiasAdd<uint8_t>::AddBiasNHWC<2>(utils::ThreadPool *thread_pool,
                                               const Tensor *input,
                                               const Tensor *bias,
                                               Tensor *output);
}  // namespace arm
}  // namespace ops
}  // namespace mace
