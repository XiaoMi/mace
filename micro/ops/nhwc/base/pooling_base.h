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

#ifndef MICRO_OPS_NHWC_BASE_POOLING_BASE_H_
#define MICRO_OPS_NHWC_BASE_POOLING_BASE_H_

#include "micro/model/output_shape.h"
#include "micro/ops/nhwc/base/filter_op_base.h"

namespace micro {
namespace ops {

enum PoolingType {
  AVG = 1,  // avg_pool
  MAX = 2,  // max_pool
};

class PoolingBase : public FilterOpBase {
 public:
  MaceStatus OnInit();
  MaceStatus Run();

 protected:
  virtual void MaxPooling(const mifloat *input, const int32_t *filter_hw,
                          const int32_t *stride_hw, const int32_t *dilation_hw,
                          const int32_t *pad_hw);
  virtual void AvgPooling(const mifloat *input, const int32_t *filter_hw,
                          const int32_t *stride_hw, const int32_t *dilation_hw,
                          const int32_t *pad_hw);

 protected:
  const mifloat *input_;
  const int32_t *input_dims_;
  uint32_t input_dim_size_;

  mifloat *output_;
  const int32_t *output_dims_;
  uint32_t output_dim_size_;

  const int32_t *kernel_;
  int32_t filter_dims_[4];
  RoundType round_type_;
  PoolingType pooling_type_;

  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
}  // namespace ops
}  // namespace micro


#endif  // MICRO_OPS_NHWC_BASE_POOLING_BASE_H_
