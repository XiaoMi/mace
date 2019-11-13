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

#ifndef MICRO_OPS_NHWC_POOLING_REF_H_
#define MICRO_OPS_NHWC_POOLING_REF_H_

#include "micro/model/output_shape.h"
#include "micro/ops/nhwc/base/pooling_base.h"

namespace micro {
namespace ops {

class PoolingRefOp : public PoolingBase {
 private:
  void MaxPooling(const mifloat *input, const int32_t *filter_hw,
                  const int32_t *stride_hw, const int32_t *dilation_hw,
                  const int32_t *pad_hw);
  void AvgPooling(const mifloat *input, const int32_t *filter_hw,
                  const int32_t *stride_hw, const int32_t *dilation_hw,
                  const int32_t *pad_hw);
};
}  // namespace ops
}  // namespace micro


#endif  // MICRO_OPS_NHWC_POOLING_REF_H_
