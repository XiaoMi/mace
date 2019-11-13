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

#include "micro/ops/expand_dims.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/scratch_buffer.h"
#include "micro/model/argument.h"

namespace micro {
namespace ops {

MaceStatus ExpandDimsOp::OnInit() {
  input_ = GetInputData<mifloat>(INPUT);
  input_dims_ = GetInputShapeDims(INPUT);
  input_dim_size_ = GetInputShapeDimSize(INPUT);

  output_ = GetOutputData<mifloat>(OUTPUT);
  axis_ = GetArgByName("axis", static_cast<int32_t>(0));
  if (axis_ < 0) {
    axis_ += input_dim_size_ + 1;
  }
  MACE_ASSERT2(axis_ >= 0 && axis_ <= static_cast<int32_t>(input_dim_size_),
               "axis is out of bound: ", axis_);

  return MACE_SUCCESS;
}

MaceStatus ExpandDimsOp::Run() {
  int32_t output_dim_size = input_dim_size_ + 1;
  int32_t *output_dims =
      ScratchBuffer(engine_config_).GetBuffer<int32_t>(output_dim_size);

  for (int32_t i = 0; i < output_dim_size; ++i) {
    if (i < axis_) {
      output_dims[i] = input_dims_[i];
    } else if (i == axis_) {
      output_dims[i] = 1;
    } else {
      output_dims[i] = input_dims_[i - 1];
    }
  }

  // TODO(luxuhui): optimize this method by reusing buffer
  int32_t input_data_size = base::GetShapeSize(input_dim_size_, input_dims_);
  base::memcpy(output_, input_, input_data_size * sizeof(mifloat));
  return ResizeOutputShape(OUTPUT, output_dim_size, output_dims);
}

}  // namespace ops
}  // namespace micro
