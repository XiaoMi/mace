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

#include "micro/ops/bias_add.h"

#include "micro/base/logging.h"
#include "micro/ops/utils/crumb_utils.h"

namespace micro {
namespace ops {

MaceStatus BiasAddOp::OnInit() {
  MACE_ASSERT1(static_cast<DataFormat>(
                   GetArgByName("data_format", static_cast<int32_t>(NHWC)))
                   != NCHW, "Now only support NHWC");

  input_ = GetInputData<mifloat>(INPUT);
  input_dims_ = GetInputShapeDims(INPUT);
  input_dim_size_ = GetInputShapeDimSize(INPUT);

  bias_ = GetInputData<mifloat>(BIAS);
  bias_dims_ = GetInputShapeDims(BIAS);
  bias_dim_size_ = GetInputShapeDimSize(BIAS);

  output_ = GetOutputData<mifloat>(OUTPUT);

  MACE_ASSERT1(bias_dim_size_ == 1, "Bias dim must be 1.");
  MACE_ASSERT1(bias_dims_[0] == input_dims_[input_dim_size_ - 1],
               "The bias's channel dim should be equal to the input's");

  return ResizeOutputShape(OUTPUT, input_dim_size_, input_dims_);
}

MaceStatus BiasAddOp::Run() {
  return crumb::ComputeBias(input_, input_dims_, input_dim_size_,
                            bias_, bias_dims_[0], output_);
}

}  // namespace ops
}  // namespace micro
