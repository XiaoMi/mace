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

#include "micro/ops/reduce.h"

#include "micro/base/logging.h"

namespace micro {
namespace ops {

MaceStatus ReduceOpBase::OnInit() {
  reduce_type_ = static_cast<ReduceType>(
      GetArgByName("reduce_type", static_cast<int32_t>(MEAN)));
  axis_ = GetRepeatArgByName<int32_t>("axis", &axis_size_);
  keep_dims_ = GetArgByName("keepdims", false);

  return MACE_SUCCESS;
}

void ReduceOpBase::Validate() {
#ifndef NDEBUG
  const int32_t input_dim_size = GetInputShapeDimSize(INPUT);
  const int32_t left = input_dim_size * -1;
  const int32_t right = input_dim_size;
  if (axis_size_) {
    for (uint32_t i = 0; i < axis_size_; ++i) {
      MACE_ASSERT1(axis_[i] > left && axis_[i] < right, "Axis is over range.");
    }
  }
#endif
}

}  // namespace ops
}  // namespace micro
