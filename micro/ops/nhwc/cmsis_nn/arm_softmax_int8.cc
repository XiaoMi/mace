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

#include "micro/ops/nhwc/cmsis_nn/arm_softmax_int8.h"

#include <arm_nnfunctions.h>

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/op_context.h"
#include "micro/model/net_def.h"
#include "micro/ops/nhwc/cmsis_nn/utilities.h"

namespace micro {
namespace ops {

MaceStatus ArmSoftmaxInt8Op::OnInit() {
  data_format_ = static_cast<DataFormat>(
      GetArgByName("data_format", static_cast<int32_t>(NHWC)));
  input_ = GetInputData<mifloat>(INPUT);
  input_dims_ = GetInputShapeDims(INPUT);
  input_dim_size_ = GetInputShapeDimSize(INPUT);
  MACE_ASSERT(input_dim_size_ == 2);

  output_ = GetOutputData<mifloat>(OUTPUT);

  bool use_log = GetArgByName("use_log", false);
  MACE_ASSERT1(!use_log, "The argument \"use_log\" is unsupported");

  return MACE_SUCCESS;
}

MaceStatus ArmSoftmaxInt8Op::Run() {
  MACE_RETURN_IF_ERROR(ResizeOutputShape(OUTPUT, input_dim_size_, input_dims_));
  // TODO(ZhangZhimin): Workarounds for AUTO data format
  if (NHWC == data_format_ || AUTO == data_format_) {  // NHWC
    return RunForNHWC();
  } else {
    MACE_NOT_IMPLEMENTED;
    return MACE_UNSUPPORTED;
  }
}

MaceStatus ArmSoftmaxInt8Op::RunForNHWC() {
  int32_t class_size = input_dims_[input_dim_size_ - 1];

  const int8_t *input_data = reinterpret_cast<const int8_t *>(input_);
  int8_t *output_data = reinterpret_cast<int8_t *>(output_);

  int32_t num_rows = input_dims_[0];

  QuantizeInfo input_quantize_info = GetInputQuantizeInfo(INPUT);

  int kInputDeltaIntBits = 5;
  int32_t scale_q = static_cast<int32_t>(
      base::min(static_cast<double>(input_quantize_info.scale) *
                    (1 << (31 - kInputDeltaIntBits)),
                (1ll << 31) - 1.0));
  int32_t mult;
  int32_t shift;
  QuantizeMultiplier(scale_q, &mult, &shift);
  int32_t diff_min = -128;

  arm_softmax_s8(input_data, num_rows, class_size, mult, shift, diff_min,
                 output_data);

  return MACE_SUCCESS;
}

}  // namespace ops
}  // namespace micro
