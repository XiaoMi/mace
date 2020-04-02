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

#include "micro/ops/activation.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/model/argument.h"

namespace micro {
namespace ops {

namespace {
template<typename T>
void PReLUActivation(const T *input_ptr, const int32_t outer_size,
                     const int32_t channel, const T *alpha_ptr,
                     T *output_ptr) {
  for (int32_t i = 0; i < outer_size; ++i) {
    const int32_t outer_base = i * channel;
    for (int32_t c = 0; c < channel; ++c) {
      const int32_t idx = outer_base + c;
      if (input_ptr[idx] < 0) {
        output_ptr[idx] = input_ptr[idx] * alpha_ptr[c];
      } else {
        output_ptr[idx] = input_ptr[idx];
      }
    }
  }
}
}  // namespace

MaceStatus ActivationOp::OnInit() {
  input_ = GetInputData<mifloat>(INPUT);
  input_dims_ = GetInputShapeDims(INPUT);
  input_dim_size_ = GetInputShapeDimSize(INPUT);
  output_ = GetOutputData<mifloat>(OUTPUT);

  return activation_.Init(this);
}

MaceStatus ActivationOp::Run() {
  MACE_RETURN_IF_ERROR(ResizeOutputShape(OUTPUT, input_dim_size_, input_dims_));
  if (activation_.GetActivationType() == PRELU) {
    MACE_ASSERT(GetInputSize() > 1);
    const mifloat *alpha = GetInputData<mifloat>(ALPHA);
    const int32_t outer_size =
        base::accumulate_multi(input_dims_, 0, input_dim_size_ - 1);
    const int32_t channel = input_dims_[input_dim_size_ - 1];
    PReLUActivation(input_, outer_size, channel, alpha, output_);
    return MACE_SUCCESS;
  } else {
    const int32_t input_size = base::GetShapeSize(input_dim_size_, input_dims_);
    return activation_.Compute(input_, input_size, output_);
  }
}

}  // namespace ops
}  // namespace micro
