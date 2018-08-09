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

#ifndef MACE_KERNELS_SOFTMAX_H_
#define MACE_KERNELS_SOFTMAX_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
#include <limits>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"
#include "mace/utils/utils.h"
#include "mace/kernels/fixpoint.h"
#include "mace/kernels/gemmlowp_util.h"
#include "mace/kernels/quantize.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

template<DeviceType D, typename T>
struct SoftmaxFunctor;

template<>
struct SoftmaxFunctor<DeviceType::CPU, float> {
  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const float *input_data = input->data<float>();
    float *output_data = output->mutable_data<float>();

    // softmax for nchw image
    if (input->dim_size() == 4) {
      const index_t batch = input->dim(0);
      const index_t class_count = input->dim(1);
      const index_t class_size = input->dim(2) * input->dim(3);
      const index_t batch_size = class_count * class_size;

      for (index_t b = 0; b < batch; ++b) {
#pragma omp parallel for
        for (index_t k = 0; k < class_size; ++k) {
          const float *input_ptr = input_data + b * batch_size + k;
          float *output_ptr = output_data + b * batch_size + k;

          float max_val = std::numeric_limits<float>::lowest();
          index_t channel_offset = 0;
          for (index_t c = 0; c < class_count; ++c) {
            float data = input_ptr[channel_offset];
            if (data > max_val) {
              max_val = data;
            }
            channel_offset += class_size;
          }

          channel_offset = 0;
          float sum = 0;
          for (index_t c = 0; c < class_count; ++c) {
            float exp_value = ::exp(input_ptr[channel_offset] - max_val);
            sum += exp_value;
            output_ptr[channel_offset] = exp_value;
            channel_offset += class_size;
          }

          sum = std::max(sum, std::numeric_limits<float>::min());
          channel_offset = 0;
          for (index_t c = 0; c < class_count; ++c) {
            output_ptr[channel_offset] /= sum;
            channel_offset += class_size;
          }
        }  // k
      }  // b
    } else if (input->dim_size() == 2) {  // normal 2d softmax
      const index_t class_size = input->dim(0);
      const index_t class_count = input->dim(1);
#pragma omp parallel for
      for (index_t k = 0; k < class_size; ++k) {
        const float *input_ptr = input_data + k * class_count;
        float *output_ptr = output_data + k * class_count;

        float max_val = std::numeric_limits<float>::lowest();
        for (index_t c = 0; c < class_count; ++c) {
          max_val = std::max(max_val, input_ptr[c]);
        }

        float sum = 0;
        for (index_t c = 0; c < class_count; ++c) {
          float exp_value = ::exp(input_ptr[c] - max_val);
          sum += exp_value;
          output_ptr[c] = exp_value;
        }

        sum = std::max(sum, std::numeric_limits<float>::min());
        for (index_t c = 0; c < class_count; ++c) {
          output_ptr[c] /= sum;
        }
      }
    } else {
      MACE_NOT_IMPLEMENTED;
    }

    return MACE_SUCCESS;
  }
};

static const int kInputDeltaIntBits = 5;
static const int kSumExpIntBits = 12;

template<>
struct SoftmaxFunctor<DeviceType::CPU, uint8_t> {
  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    // Ignore range stat, fix range to [0, 1]. For large depth, each softmax
    // output may be too small (<<1), which causes precision issue. But it is
    // fine when doing classification inference.
    output->SetScale(1.f / 255);
    output->SetZeroPoint(0);

    using FixPointInputDelta = gemmlowp::FixedPoint<int32_t,
                                                    kInputDeltaIntBits>;
    using FixPointSumExp = gemmlowp::FixedPoint<int32_t, kSumExpIntBits>;
    using FixPoint0 = gemmlowp::FixedPoint<int32_t, 0>;

    MACE_CHECK(input->dim_size() == 2 || input->dim_size() == 4,
               "Softmax does not support dim size: ",
               input->dim_size());
    index_t batch;
    index_t depth;

    if (input->dim_size() == 2) {
      batch = input->dim(0);
      depth = input->dim(1);
    } else {
      batch = input->dim(0) * input->dim(1) * input->dim(2);
      depth = input->dim(3);
    }

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const uint8_t *input_data = input->data<uint8_t>();
    float input_scale = input->scale();
    uint8_t *output_data = output->mutable_data<uint8_t>();

    // If depth is short, do it using float32. Float computation should not
    // be here, but as long as it is on CPU, it is fine.
    if (depth < 32) {
#pragma omp parallel for
      for (index_t b = 0; b < batch; ++b) {
        const uint8_t *input_ptr = input_data + b * depth;
        uint8_t *output_ptr = output_data + b * depth;

        uint8_t max_value = FindMax(input_ptr, depth);
        float sum = 0;
        std::vector<float> depth_cache(depth);
        for (index_t d = 0; d < depth; ++d) {
          float exp_value = ::exp((static_cast<int>(input_ptr[d]) - max_value)
                                      * input_scale);
          sum += exp_value;
          depth_cache[d] = exp_value;
        }

        sum = std::max(sum, std::numeric_limits<float>::min());
        for (index_t d = 0; d < depth; ++d) {
          double output_f = depth_cache[d] / sum;
          output_ptr[d] = static_cast<uint8_t>(output_f * 255);
        }
      }
      return MACE_SUCCESS;
    }

    int32_t scale_q = static_cast<int32_t>(std::min(
        static_cast<double>(input_scale) * (1 << (31 - kInputDeltaIntBits)),
        (1ll << 31) - 1.0));
    int32_t input_delta_limit = -((1ll << 31) - 1) / scale_q;

#pragma omp parallel for
    for (index_t b = 0; b < batch; ++b) {
      const uint8_t *input_ptr = input_data + b * depth;
      uint8_t *output_ptr = output_data + b * depth;

      FixPointSumExp sum = FixPointSumExp::Zero();
      uint8_t max_value = FindMax(input_ptr, depth);
      index_t d = 0;

      // Neon optimization is not useful so far as we benchmark.
      // Enable it when we find a case that proves it useful.
#if 0 && defined(MACE_ENABLE_NEON)
      using FixPointInputDeltaInt32x4 = gemmlowp::FixedPoint<int32x4_t,
                                                         kInputDeltaIntBits>;
      using FixPointSumExpInt32x4 = gemmlowp::FixedPoint<int32x4_t,
                                                         kSumExpIntBits>;
      using FixPoint0Int32x4 = gemmlowp::FixedPoint<int32x4_t, 0>;

      int16x8_t vmax_value_s16 = vdupq_n_s16(max_value);
      int32x4_t vinput_delta_limit_s32 = vdupq_n_s32(input_delta_limit);

      FixPointSumExpInt32x4 vsum_s32_fp_0 = FixPointSumExpInt32x4::Zero();
      FixPointSumExpInt32x4 vsum_s32_fp_1 = FixPointSumExpInt32x4::Zero();
      FixPointSumExpInt32x4 vzero_s32_fp = FixPointSumExpInt32x4::Zero();

      int32_t scale_q_multipler, scale_q_shift;
      QuantizeMultiplier(scale_q, &scale_q_multipler, &scale_q_shift);
      FixPointInputDeltaInt32x4 vscale_s32_fp =
          FixPointInputDeltaInt32x4::FromScalarRaw(scale_q);
      FixPoint0Int32x4 vscale_s32_fp_multiplier =
          FixPoint0Int32x4::FromScalarRaw(scale_q_multipler);

      for (; d <= depth - 8; d += 8) {
        uint16x8_t vinput_u16 = vmovl_u8(vld1_u8(input_ptr + d));
        int16x8_t vinput_delta_s16 =
            vsubq_s16(vreinterpretq_s16_u16(vinput_u16), vmax_value_s16);
        int32x4_t input_delta_s32_0 = vmovl_s16(vget_low_s16(vinput_delta_s16));
        int32x4_t
            input_delta_s32_1 = vmovl_s16(vget_high_s16(vinput_delta_s16));
        int32x4_t vmask_s32_0 =
            gemmlowp::MaskIfGreaterThanOrEqual(input_delta_s32_0,
                                               vinput_delta_limit_s32);
        int32x4_t vmask_s32_1 =
            gemmlowp::MaskIfGreaterThanOrEqual(input_delta_s32_1,
                                               vinput_delta_limit_s32);
        FixPointInputDeltaInt32x4
            vscaled_input_delta_s32_fp_0 = vscale_s32_fp_multiplier *
            FixPointInputDeltaInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_delta_s32_0, scale_q_shift));
        FixPointInputDeltaInt32x4
            vscaled_input_delta_s32_fp_1 = vscale_s32_fp_multiplier *
            FixPointInputDeltaInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_delta_s32_1, scale_q_shift));
        FixPointSumExpInt32x4 vexp_s32_fp_0 = gemmlowp::Rescale<kSumExpIntBits>(
            exp_on_negative_values(vscaled_input_delta_s32_fp_0));
        FixPointSumExpInt32x4 vexp_s32_fp_1 = gemmlowp::Rescale<kSumExpIntBits>(
            exp_on_negative_values(vscaled_input_delta_s32_fp_1));
        FixPointSumExpInt32x4 vmasked_exp_s32_fp_0 =
            SelectUsingMask(vmask_s32_0, vexp_s32_fp_0, vzero_s32_fp);
        FixPointSumExpInt32x4 vmasked_exp_s32_fp_1 =
            SelectUsingMask(vmask_s32_1, vexp_s32_fp_1, vzero_s32_fp);
        vsum_s32_fp_0 = vsum_s32_fp_0 + vmasked_exp_s32_fp_0;
        vsum_s32_fp_1 = vsum_s32_fp_1 + vmasked_exp_s32_fp_1;
      }
      int32x4_t vsum_s32 = (vsum_s32_fp_0 + vsum_s32_fp_1).raw();
      int32x2_t vsum_reduced_2_s32 =
          vadd_s32(vget_low_s32(vsum_s32), vget_high_s32(vsum_s32));
      int32x2_t vsum_reduced_1_s32 =
          vpadd_s32(vsum_reduced_2_s32, vsum_reduced_2_s32);
      sum = FixPointSumExp::FromRaw(vget_lane_s32(vsum_reduced_1_s32, 0));
#endif
      for (; d < depth; ++d) {
        int32_t input_delta = static_cast<int32_t>(input_ptr[d]) - max_value;
        if (input_delta >= input_delta_limit) {
          int32_t scaled_input_delta_q = scale_q * input_delta;
          FixPointInputDelta scaled_input_delta_fp =
              FixPointInputDelta::FromRaw(scaled_input_delta_q);
          sum = sum + gemmlowp::Rescale<kSumExpIntBits>(
              exp_on_negative_values(scaled_input_delta_fp));
        }
      }

      int32_t sum_q = sum.raw();
      int left_zero_count =
          __builtin_clz(static_cast<uint32_t>(sum_q));
      int tail_count = kSumExpIntBits - left_zero_count;
      int32_t fractional_q0 = static_cast<int32_t>(
          (static_cast<uint32_t>(sum_q) << left_zero_count) -
              (static_cast<uint32_t>(1) << 31));
      FixPoint0 recip_sum_q0 = gemmlowp::one_over_one_plus_x_for_x_in_0_1(
          FixPoint0::FromRaw(fractional_q0));

      d = 0;

      // Neon optimization is not useful so far as we benchmark.
      // Enable it when we find a case that proves it useful.
#if 0 && defined(MACE_ENABLE_NEON)
      FixPoint0Int32x4 vrecip_sum_q0_s32_fp =
          FixPoint0Int32x4::FromScalarRaw(recip_sum_q0.raw());
      int16x8_t vinput_delta_limit_s16 = vdupq_n_s16(input_delta_limit);
      for (; d <= depth - 8; d += 8) {
        uint16x8_t vinput_u16 = vmovl_u8(vld1_u8(input_ptr + d));
        int16x8_t vinput_delta_s16 =
            vsubq_s16(vreinterpretq_s16_u16(vinput_u16), vmax_value_s16);
        int32x4_t input_delta_s32_0 = vmovl_s16(vget_low_s16(vinput_delta_s16));
        int32x4_t
            input_delta_s32_1 = vmovl_s16(vget_high_s16(vinput_delta_s16));
        int16x8_t vmask_s16 = gemmlowp::MaskIfGreaterThanOrEqual(
            vinput_delta_s16,
            vinput_delta_limit_s16);
        FixPointInputDeltaInt32x4
            vscaled_input_delta_s32_fp_0 = vscale_s32_fp_multiplier *
            FixPointInputDeltaInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_delta_s32_0, scale_q_shift));
        FixPointInputDeltaInt32x4
            vscaled_input_delta_s32_fp_1 = vscale_s32_fp_multiplier *
            FixPointInputDeltaInt32x4::FromRaw(
                gemmlowp::ShiftLeft(input_delta_s32_1, scale_q_shift));
        FixPoint0Int32x4 vexp_s32_fp_0 =
            exp_on_negative_values(vscaled_input_delta_s32_fp_0);
        FixPoint0Int32x4 vexp_s32_fp_1 =
            exp_on_negative_values(vscaled_input_delta_s32_fp_1);
        int32x4_t voutput_data_s32_0 = gemmlowp::RoundingDivideByPOT(
            (vrecip_sum_q0_s32_fp * vexp_s32_fp_0).raw(), tail_count + 31 - 8);
        int32x4_t voutput_data_s32_1 = gemmlowp::RoundingDivideByPOT(
            (vrecip_sum_q0_s32_fp * vexp_s32_fp_1).raw(), tail_count + 31 - 8);
        int16x8_t voutput_data_s16 =
            vcombine_s16(vqmovn_s32(voutput_data_s32_0),
                         vqmovn_s32(voutput_data_s32_1));
        int16x8_t masked_voutput_data_s16 =
            gemmlowp::SelectUsingMask(vmask_s16,
                                      voutput_data_s16,
                                      vdupq_n_s16(0));
        uint8x8_t voutput_u8 = vqmovun_s16(masked_voutput_data_s16);
        vst1_u8(output_ptr + d, voutput_u8);
      }
#endif
      for (; d < depth; ++d) {
        int32_t input_delta = static_cast<int32_t>(input_ptr[d]) - max_value;
        if (input_delta >= input_delta_limit) {
          int32_t scaled_input_delta_q = scale_q * input_delta;
          FixPointInputDelta scaled_input_delta_fp =
              FixPointInputDelta::FromRaw(scaled_input_delta_q);

          FixPoint0 exp = exp_on_negative_values(scaled_input_delta_fp);
          int32_t output_data = gemmlowp::RoundingDivideByPOT(
              (recip_sum_q0 * exp).raw(), tail_count + 31 - 8);

          output_ptr[d] = std::max(std::min(output_data, 255), 0);
        }
      }
    }

    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template<typename T>
struct SoftmaxFunctor<DeviceType::GPU, T> {
  MaceStatus operator()(const Tensor *logits,
                        Tensor *output,
                        StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_SOFTMAX_H_
