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

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"

#ifdef MACE_ENABLE_QUANTIZE
#include "mace/ops/fixpoint.h"
#include "mace/ops/common/gemmlowp_util.h"
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/softmax.h"
#include "mace/ops/opencl/buffer/softmax.h"
#endif  // MACE_ENABLE_OPENCL

#include "mace/utils/memory.h"

namespace mace {
namespace ops {

template<DeviceType D, typename T>
class SoftmaxOp;

template<class T>
class SoftmaxOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit SoftmaxOp(OpConstructContext *context)
      : Operation(context),
        use_log_(Operation::GetOptionalArg<bool>("use_log", false)),
        has_df_(Operation::GetOptionalArg<int>("has_data_format", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);

    if (isNCHW(input)) {  // NCHW
      return RunForNCHW(context);
    } else {
      return RunForNHWC(context);
    }
  }

 protected:
  bool use_log_;
  bool has_df_;
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);

 protected:
  MaceStatus RunForNCHW(OpContext *context) {
    const Tensor *input = this->Input(INPUT);
    const T *input_data = input->data<T>();
    Tensor *output = this->Output(OUTPUT);
    T *output_data = output->mutable_data<T>();

    MACE_CHECK(input->dim_size() == 4, "The dim size of NCHW should be 4.");
    index_t hw_stride = input->dim(3);
    index_t hw_size = hw_stride * input->dim(2);
    index_t class_stride = hw_size;
    index_t class_size = class_stride * input->dim(1);
    index_t batch_stride = class_size;
    index_t batch_size = batch_stride * input->dim(0);

    auto cache_buffer = context->device()->scratch_buffer();
    cache_buffer->Rewind();
    MACE_RETURN_IF_ERROR(cache_buffer->GrowSize(hw_size * sizeof(float)));
    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();
    float std_lowest = std::numeric_limits<float>::lowest();
    float *cache_ptr = cache_buffer->mutable_data<float>();

    for (index_t b_offset = 0;
         b_offset < batch_size; b_offset += batch_stride) {
      const T *input_b_base = input_data + b_offset;
      T *output_b_base = output_data + b_offset;
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        const auto raw_step_size = step * sizeof(float);
        for (index_t k = start; k < end; k += step) {
          float *cache_k_ptr = cache_ptr + k;
          for (index_t i = 0; i < step; ++i) {
            cache_k_ptr[i] = std_lowest;
          }
        }

        for (index_t c_offset = 0; c_offset < class_size;
             c_offset += class_stride) {
          const T *input_c_base = input_b_base + c_offset;
          for (index_t k = start; k < end; k += step) {
            const T *input_ptr = input_c_base + k;
            float *cache_k_ptr = cache_ptr + k;
            for (index_t i = 0; i < step; ++i) {
              cache_k_ptr[i] = std::max(cache_k_ptr[i], input_ptr[i]);
            }
          }
        }

        for (index_t c_offset = 0; c_offset < class_size;
             c_offset += class_stride) {
          const T *input_c_base = input_b_base + c_offset;
          T *output_c_base = output_b_base + c_offset;
          for (index_t k = start; k < end; k += step) {
            const T *input_ptr = input_c_base + k;
            T *output_ptr = output_c_base + k;
            float *cache_k_ptr = cache_ptr + k;
            for (index_t i = 0; i < step; ++i) {
              output_ptr[i] = std::exp(input_ptr[i] - cache_k_ptr[i]);
            }
          }
        }

        for (index_t k = start; k < end; k += step) {
          memset(cache_ptr + k, 0, raw_step_size);
        }

        for (index_t c_offset = 0; c_offset < class_size;
             c_offset += class_stride) {
          T *output_c_base = output_b_base + c_offset;
          for (index_t k = start; k < end; k += step) {
            T *output_ptr = output_c_base + k;
            float *cache_k_ptr = cache_ptr + k;
            for (index_t i = 0; i < step; ++i) {
              cache_k_ptr[i] += static_cast<float>(output_ptr[i]);
            }
          }
        }

        for (index_t c_offset = 0; c_offset < class_size;
             c_offset += class_stride) {
          T *output_c_base = output_b_base + c_offset;
          for (index_t k = start; k < end; k += step) {
            T *output_ptr = output_c_base + k;
            float *cache_k_ptr = cache_ptr + k;
            for (index_t i = 0; i < step; ++i) {
              output_ptr[i] /= cache_k_ptr[i];
            }
          }
        }

        if (use_log_) {
          for (index_t c_offset = 0; c_offset < class_size;
               c_offset += class_stride) {
            T *output_c_base = output_b_base + c_offset;
            for (index_t k = start; k < end; k += step) {
              T *output_ptr = output_c_base + k;
              for (index_t i = 0; i < step; ++i) {
                output_ptr[i] = std::log(output_ptr[i]);
              }
            }
          }
        }  // use_log_
      }, 0, hw_size, hw_stride);
    }

    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus RunForNHWC(OpContext *context) {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    T *output_data = output->mutable_data<T>();

    MACE_CHECK(input->dim_size() >= 2, "The input->dim_size() >= 2 failed.");
    index_t class_size = input->dim(input->dim_size() - 1);
    index_t hw_stride = class_size;
    index_t hw_size = std::accumulate(input->shape().begin() + 1,
                                      input->shape().end() - 1,
                                      hw_stride,
                                      std::multiplies<index_t>());
    index_t batch_stride = hw_size;
    index_t batch_size = std::accumulate(input->shape().begin(),
                                         input->shape().end(),
                                         1,
                                         std::multiplies<index_t>());

    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();
    const T *input_data = input->data<T>();
    float std_lowest = std::numeric_limits<float>::lowest();
    for (index_t b_offset = 0; b_offset < batch_size;
         b_offset += batch_stride) {
      const T *input_b_ptr = input_data + b_offset;
      T *output_b_ptr = output_data + b_offset;
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t k = start; k < end; k += step) {
          const T *input_ptr = input_b_ptr + k;
          T *output_ptr = output_b_ptr + k;

          float max_val = std_lowest;
          for (index_t c = 0; c < class_size; ++c) {
            max_val = std::max(max_val, input_ptr[c]);
          }

          float sum = 0;
          for (index_t c = 0; c < class_size; ++c) {
            float exp_value = std::exp(input_ptr[c] - max_val);
            sum += exp_value;
            output_ptr[c] = exp_value;
          }

          if (use_log_) {
            for (index_t c = 0; c < class_size; ++c) {
              float output = (static_cast<float>(output_ptr[c])) / sum;
              output_ptr[c] = std::log(output);
            }
          } else {
            for (index_t c = 0; c < class_size; ++c) {
              output_ptr[c] /= sum;
            }
          }
        }  // k
      }, 0, hw_size, hw_stride);
    }  // b_offset

    return MaceStatus::MACE_SUCCESS;
  }

  inline bool isNCHW(const Tensor *input) {
    auto data_format = input->data_format();
    auto dim_size = input->dim_size();
    return dim_size == 4 && (has_df_ || data_format == DataFormat::NCHW);
  }
};

#ifdef MACE_ENABLE_QUANTIZE
static const int kInputDeltaIntBits = 6;
static const int kSumExpIntBits = 12;

template <>
class SoftmaxOp<DeviceType::CPU, uint8_t> : public Operation {
 public:
  explicit SoftmaxOp(OpConstructContext *context)
      : Operation(context),
        use_log_(Operation::GetOptionalArg<bool>("use_log", false)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    MACE_CHECK(!use_log_, "MACE dose not support quantized logsoftmax yet.");
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
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

    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();

    // If depth is short, do it using float32. Float computation should not
    // be here, but as long as it is on CPU, it is fine.
    if (depth < 32) {
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t b = start; b < end; b += step) {
          const uint8_t *input_ptr = input_data + b * depth;
          uint8_t *output_ptr = output_data + b * depth;

          uint8_t max_value = FindMax(input_ptr, depth);
          float sum = 0;
          std::vector<float> depth_cache(depth);
          for (index_t d = 0; d < depth; ++d) {
            float exp_value = std::exp(
                (static_cast<int>(input_ptr[d]) - max_value) * input_scale);
            sum += exp_value;
            depth_cache[d] = exp_value;
          }

          sum = std::max(sum, std::numeric_limits<float>::min());
          for (index_t d = 0; d < depth; ++d) {
            double output_f = depth_cache[d] / sum;
            output_ptr[d] = static_cast<uint8_t>(output_f * 255);
          }
        }
      }, 0, batch, 1);

      return MaceStatus::MACE_SUCCESS;
    }

    int32_t scale_q = static_cast<int32_t>(std::min(
        static_cast<double>(input_scale) * (1 << (31 - kInputDeltaIntBits)),
        (1ll << 31) - 1.0));
    int32_t input_delta_limit = -((1ll << 31) - 1) / scale_q;

    thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
      for (index_t b = start; b < end; b += step) {
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
    }, 0, batch, 1);

    return MaceStatus::MACE_SUCCESS;
  }

 protected:
  bool use_log_;
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
template<>
class SoftmaxOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit SoftmaxOp(OpConstructContext *context)
      : Operation(context) {
    bool use_log = (
        Operation::GetOptionalArg<bool>("use_log", false));
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::SoftmaxKernel>(use_log);
    } else {
      kernel_ = make_unique<opencl::buffer::SoftmaxKernel>(use_log);
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    return kernel_->Compute(context, input, output);
  }

 private:
  std::unique_ptr<OpenCLSoftmaxKernel> kernel_;
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif  // MACE_ENABLE_OPENCL

void RegisterSoftmax(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "Softmax", SoftmaxOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "Softmax", SoftmaxOp,
                        DeviceType::CPU);

#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "Softmax", SoftmaxOp,
                   DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE

  MACE_REGISTER_GPU_OP(op_registry, "Softmax", SoftmaxOp);

  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("Softmax")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<DeviceType> {
                auto op = context->operator_def();
                if (op->output_shape_size() != op->output_size()) {
                  return {DeviceType::CPU, DeviceType::GPU};
                }
                if (op->output_shape(0).dims_size() != 2 &&
                    op->output_shape(0).dims_size() != 4) {
                  return {DeviceType::CPU};
                }
                return {DeviceType::CPU, DeviceType::GPU};
              }));
}

}  // namespace ops
}  // namespace mace
