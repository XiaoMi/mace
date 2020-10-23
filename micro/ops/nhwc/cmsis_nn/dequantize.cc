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

#include "micro/ops/nhwc/cmsis_nn/dequantize.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/op_context.h"
#include "micro/framework/operator.h"
#include "micro/model/net_def.h"

namespace micro {
namespace ops {

MaceStatus DequantizeOp::OnInit() {
  input_ = GetInputData<int8_t>(INPUT);
  input_dims_ = GetInputShapeDims(INPUT);
  input_dim_size_ = GetInputShapeDimSize(INPUT);

  output_ = GetOutputData<mifloat>(OUTPUT);

  return MACE_SUCCESS;
}

MaceStatus DequantizeOp::Run() {
  MACE_RETURN_IF_ERROR(ResizeOutputShape(OUTPUT, input_dim_size_, input_dims_));

  QuantizeInfo input_quantize_info = GetInputQuantizeInfo(INPUT);

  float scale = input_quantize_info.scale;
  int32_t zero_point = input_quantize_info.zero;

  int32_t element_size = 1;
  for (uint32_t i = 0; i < input_dim_size_; ++i) {
    element_size *= input_dims_[i];
  }
  for (int32_t i = 0; i < element_size; ++i) {
    output_[i] = scale * (input_[i] - zero_point);
  }

  return MACE_SUCCESS;
}

}  // namespace ops
}  // namespace micro
