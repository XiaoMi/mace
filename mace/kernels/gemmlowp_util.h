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

#ifndef MACE_KERNELS_GEMMLOWP_UTIL_H_
#define MACE_KERNELS_GEMMLOWP_UTIL_H_

#include <tuple>

#include "public/gemmlowp.h"
#include "mace/kernels/quantize.h"

namespace mace {

gemmlowp::GemmContext& GetGemmlowpContext();

struct GemmlowpOutputPipeline {
  typedef gemmlowp::VectorMap<const int32_t, gemmlowp::VectorShape::Col>
      ColVectorMap;
  typedef std::tuple<
      gemmlowp::OutputStageBiasAddition<ColVectorMap>,
      gemmlowp::OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint,
      gemmlowp::OutputStageSaturatingCastToUint8> Pipeline;
  typedef std::tuple<
      gemmlowp::OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint,
      gemmlowp::OutputStageSaturatingCastToUint8> NoBiasPipeline;

  static Pipeline Make(
      const int32_t *bias_data, const index_t channels, const float lhs_scale,
      const float rhs_scale, const float output_scale,
      const int32_t output_zero_point) {
    ColVectorMap bias_vector(bias_data, channels);
    gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
    bias_addition_stage.bias_vector = bias_vector;
    int32_t quantized_multiplier;
    int32_t right_shift;
    kernels::GetOutputMultiplierAndShift(lhs_scale, rhs_scale, output_scale,
                                         &quantized_multiplier, &right_shift);
    gemmlowp::OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint
        quantize_down_stage;
    quantize_down_stage.result_offset_after_shift = output_zero_point;
    quantize_down_stage.result_fixedpoint_multiplier = quantized_multiplier;
    quantize_down_stage.result_shift = right_shift;

    gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
    return std::make_tuple(bias_addition_stage, quantize_down_stage,
                           saturating_cast_stage);
  }

  static NoBiasPipeline MakeNoBias(
      const float lhs_scale, const float rhs_scale, const float output_scale,
      const int32_t output_zero_point) {
    int32_t quantized_multiplier;
    int32_t right_shift;
    kernels::GetOutputMultiplierAndShift(lhs_scale, rhs_scale, output_scale,
                                         &quantized_multiplier, &right_shift);
    gemmlowp::OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint
        quantize_down_stage;
    quantize_down_stage.result_offset_after_shift = output_zero_point;
    quantize_down_stage.result_fixedpoint_multiplier = quantized_multiplier;
    quantize_down_stage.result_shift = right_shift;

    gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
    return std::make_tuple(quantize_down_stage, saturating_cast_stage);
  }
};
}  // namespace mace

#endif  // MACE_KERNELS_GEMMLOWP_UTIL_H_
