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

#include "micro/ops/squeeze.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/scratch_buffer.h"

namespace micro {
namespace ops {

MaceStatus SqueezeOp::OnInit() {
  input_ = GetInputData<mifloat>(INPUT);
  input_dims_ = GetInputShapeDims(INPUT);
  input_dim_size_ = GetInputShapeDimSize(INPUT);
  MACE_ASSERT1(input_dim_size_ >= 2, "The input->dim_size() >= 2 failed.");

  output_ = GetOutputData<mifloat>(OUTPUT);

  const int32_t *axis = GetRepeatArgByName<int32_t>("axis", &axis_size_);
  data_format_ = static_cast<DataFormat>(GetArgByName(
      "data_format", static_cast<int32_t>(NHWC)));
  ScratchBuffer scratch_buffer(engine_config_);
  if (data_format_ == NCHW && input_dim_size_ == 4
      && axis_size_ == 2 && axis[0] == 1 && axis[1] == 2) {
    axis_ = scratch_buffer.GetBuffer<int32_t>(axis_size_);
    base::memcpy(axis_, axis, axis_size_ * sizeof(int32_t));
    axis_[0] = 2;
    axis_[1] = 3;
  } else {
    axis_ = const_cast<int32_t *>(axis);
  }
  resize_shape_ = scratch_buffer.GetBuffer<int32_t>(input_dim_size_);

  return MACE_SUCCESS;
}

MaceStatus SqueezeOp::Run() {
  int32_t resize_shape_idx = 0;
  for (uint32_t i = 0; i < input_dim_size_; ++i) {
    if (input_dims_[i] > 1) {
      resize_shape_[resize_shape_idx++] = input_dims_[i];
    } else if (axis_size_ > 0) {
      bool exist_in_axis = false;
      for (uint32_t k = 0; k < axis_size_; ++k) {
        if (i == static_cast<uint32_t>(axis_[k])) {
          exist_in_axis = true;
          break;
        }
      }
      if (!exist_in_axis) {
        resize_shape_[resize_shape_idx++] = input_dims_[i];
      }
    }
  }

  // TODO(luxuhui): optimize this method by reusing buffer
  const int32_t input_size = base::GetShapeSize(input_dim_size_, input_dims_);
  base::memcpy(output_, input_, input_size * sizeof(mifloat));

  return ResizeOutputShape(OUTPUT, resize_shape_idx, resize_shape_);
}

}  // namespace ops
}  // namespace micro
