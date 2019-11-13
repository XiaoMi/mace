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

#include "micro/ops/shape.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"

namespace micro {
namespace ops {

MaceStatus ShapeOp::OnInit() {
  input_dims_ = GetInputShapeDims(INPUT);
  input_dim_size_ = GetInputShapeDimSize(INPUT);
  output_ = GetOutputData<int32_t>(OUTPUT);

  return MACE_SUCCESS;
}

MaceStatus ShapeOp::Run() {
  if (input_dim_size_ > 0) {
    const int32_t out_put_dims[1] = {static_cast<int32_t>(input_dim_size_)};
    MACE_RETURN_IF_ERROR(ResizeOutputShape(OUTPUT, 1, out_put_dims));
  } else {
    ResizeOutputShape(OUTPUT, 0, NULL);
  }

  for (uint32_t i = 0; i < input_dim_size_; ++i) {
    output_[i] = static_cast<int32_t>(input_dims_[i]);
  }

  return MACE_SUCCESS;
}

}  // namespace ops
}  // namespace micro
