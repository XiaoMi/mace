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

#include "micro/ops/nhwc/base/depthwise_conv_2d_base.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/model/operator_def.h"
#include "micro/ops/utils/crumb_utils.h"

namespace micro {
namespace ops {

MaceStatus DepthwiseConv2dBase::Run() {
  int32_t output_dims[4] = {0};
  InitPaddingAndOutputSize(input_dims_, filter_dims_, FLOOR, output_dims);
  output_dims[3] *= input_dims_[3];
  ResizeOutputShape(0, 4, output_dims);

  MACE_RETURN_IF_ERROR(Compute(output_dims));

  if (bias_ != NULL) {
    MACE_RETURN_IF_ERROR(crumb::ComputeBias(
        output_, output_dims, input_dim_size_, bias_, bias_dims_[0], output_));
  }
  MACE_RETURN_IF_ERROR(activation_.Compute(
      output_, base::GetShapeSize(input_dim_size_, output_dims), output_));

  return MACE_SUCCESS;
}

}  // namespace ops
}  // namespace micro
