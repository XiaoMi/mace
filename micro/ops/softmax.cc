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

#include "micro/ops/softmax.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"

namespace micro {
namespace ops {

MaceStatus SoftmaxOp::OnInit() {
  data_format_ = static_cast<DataFormat>(GetArgByName(
      "data_format", static_cast<int32_t>(NHWC)));
  input_ = GetInputData<mifloat>(INPUT);
  input_dims_ = GetInputShapeDims(INPUT);
  input_dim_size_ = GetInputShapeDimSize(INPUT);
  MACE_ASSERT1(input_dim_size_ >= 2, "The input->dim_size() >= 2 failed.");

  output_ = GetOutputData<mifloat>(OUTPUT);
  use_log_ = GetArgByName("use_log", false);

  return MACE_SUCCESS;
}

MaceStatus SoftmaxOp::Run() {
  MACE_RETURN_IF_ERROR(ResizeOutputShape(OUTPUT, input_dim_size_, input_dims_));
  if (NHWC == data_format_) {  // NHWC
    return RunForNHWC();
  } else {
    MACE_NOT_IMPLEMENTED;
    return MACE_UNSUPPORTED;
  }
}

MaceStatus SoftmaxOp::RunForNHWC() {
  int32_t class_size = input_dims_[input_dim_size_ - 1];
  int32_t hw_stride = class_size;
  int32_t hw_size = base::accumulate_multi(input_dims_, 1, input_dim_size_);
  int32_t batch_stride = hw_size;
  int32_t batch_size = base::GetShapeSize(input_dim_size_, input_dims_);

  float std_lowest = base::lowest();
  for (int32_t b_offset = 0; b_offset < batch_size; b_offset += batch_stride) {
    const mifloat *input_b_ptr = input_ + b_offset;
    mifloat *output_b_ptr = output_ + b_offset;
    for (int32_t k = 0; k < hw_size; k += hw_stride) {
      const mifloat *input_ptr = input_b_ptr + k;
      mifloat *output_ptr = output_b_ptr + k;

      float max_val = std_lowest;
      for (int32_t c = 0; c < class_size; ++c) {
        max_val = base::max<float>(max_val, input_ptr[c]);  // NOLINT
      }

      float sum = 0;
      for (int32_t c = 0; c < class_size; ++c) {
        float exp_value = base::exp(input_ptr[c] - max_val);
        sum += exp_value;
        output_ptr[c] = exp_value;
      }

      if (use_log_) {
        for (int32_t c = 0; c < class_size; ++c) {
          float output_value = output_ptr[c];
          output_value /= sum;
          output_ptr[c] = base::log(output_value);
        }
      } else {
        for (int32_t c = 0; c < class_size; ++c) {
          output_ptr[c] = output_ptr[c] / sum;
        }
      }
    }  // k
  }  // b_offset
  return MACE_SUCCESS;
}

}  // namespace ops
}  // namespace micro
