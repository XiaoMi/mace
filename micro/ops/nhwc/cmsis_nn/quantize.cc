// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "micro/ops/nhwc/cmsis_nn/quantize.h"

#include <cmath>
#include "micro/base/logging.h"
#include "micro/base/utils.h"

namespace micro {
namespace ops {

inline int8_t SaturateInt8(float value) {
  int rounded_value = static_cast<int>(value);
  if (rounded_value <= -128) {
    return -128;
  } else if (rounded_value >= 127) {
    return 127;
  } else {
    return static_cast<int8_t>(rounded_value);
  }
}

MaceStatus QuantizeOp::OnInit() {
  input_ = GetInputData<mifloat>(INPUT);
  input_dims_ = GetInputShapeDims(INPUT);
  input_dim_size_ = GetInputShapeDimSize(INPUT);

  output_ = GetOutputData<int8_t>(OUTPUT);

  return MACE_SUCCESS;
}

MaceStatus QuantizeOp::Run() {
  MACE_RETURN_IF_ERROR(ResizeOutputShape(OUTPUT, input_dim_size_, input_dims_));
  QuantizeInfo output_quantize_info = GetOutputQuantizeInfo(OUTPUT);
  float recip_scale = 1.0f / output_quantize_info.scale;
  int32_t zero_point = output_quantize_info.zero;

  int32_t element_size = 1;
  for (uint32_t i = 0; i < input_dim_size_; ++i) {
    element_size *= input_dims_[i];
  }

  for (int32_t i = 0; i < element_size; ++i) {
    output_[i] = SaturateInt8(roundf(recip_scale * input_[i] + zero_point));
  }

  return MACE_SUCCESS;
}

}  // namespace ops
}  // namespace micro
