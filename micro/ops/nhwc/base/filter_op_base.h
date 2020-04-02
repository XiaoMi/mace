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

#ifndef MICRO_OPS_NHWC_BASE_FILTER_OP_BASE_H_
#define MICRO_OPS_NHWC_BASE_FILTER_OP_BASE_H_

#include "micro/framework/operator.h"

namespace micro {
namespace ops {

enum Padding {
  VALID = 0,  // No padding
  SAME = 1,   // Pads with half the filter size (rounded down) on both sides
  FULL = 2,   // Pads with one less than the filter size on both sides

  NONE,
};

enum RoundType {
  FLOOR = 0,
  CEIL = 1,
};

class FilterOpBase : public framework::Operator {
 public:
  MaceStatus OnInitBase();

 protected:
  void InitPaddingAndOutputSize(const int32_t *input_dims,
                                const int32_t *filter_dims,
                                const RoundType round_type,
                                int32_t *output_dims);

 private:
  void CalcPaddingAndOutputSize(const int32_t *input_dims,
                                const int32_t *filter_dims,
                                int32_t *output_dims);
  void CalcOutputSizeWithPaddingSize(const int32_t *input_dims,
                                     const int32_t *filter_dims,
                                     const RoundType round_type,
                                     int32_t *output_dims);

 protected:
  Padding padding_type_;
  const int32_t *strides_;
  int32_t padding_sizes_[2];
  int32_t dilations_[2];
};

}  // namespace ops
}  // namespace micro


#endif  // MICRO_OPS_NHWC_BASE_FILTER_OP_BASE_H_
