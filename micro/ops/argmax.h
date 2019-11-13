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

#ifndef MICRO_OPS_ARGMAX_H_
#define MICRO_OPS_ARGMAX_H_

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/operator.h"
#include "micro/framework/scratch_buffer.h"
#include "micro/include/utils/macros.h"

namespace micro {
namespace ops {

template<class T>
class ArgMaxOp : public framework::Operator {
 public:
  MaceStatus OnInit() {
    axis_ = GetArgByName("axis", static_cast<int32_t>(0));
    keep_dims_ = GetArgByName("keepdims", true);
    MACE_ASSERT1(keep_dims_, "Mace only supports keep_dims ArgMax.");
    argmin_ = GetArgByName("argmin", false);
    input_ = GetInputData<T>(INPUT);
    input_dims_ = GetInputShapeDims(INPUT);
    input_dim_size_ = GetInputShapeDimSize(INPUT);
    MACE_ASSERT1(input_dim_size_ > 0, "ArgMax input should not be a scalar");

    output_ = GetOutputData<int32_t>(OUTPUT);
    output_dims_ = GetOutputShapeDims(OUTPUT);
    output_dim_size_ = GetOutputShapeDimSize(OUTPUT);
    return MACE_SUCCESS;
  }

  MaceStatus Run() {
    int32_t axis_value = 0;
    const int32_t *axis = GetInputSize() == 2 ?
                          GetInputData<int32_t>(AXIS) : NULL;
    if (axis != NULL) {
      MACE_ASSERT1(GetInputShapeDimSize(AXIS) == 0,
                   "Mace argmax only supports scalar axis");
      axis_value = axis[0];
    } else {
      axis_value = axis_;
    }
    if (axis_value < 0) {
      axis_value += input_dim_size_;
    }
    MACE_ASSERT1(axis_value == static_cast<int32_t>(input_dim_size_) - 1,
                 "Mace argmax only supports last dimension as axis");

    MACE_ASSERT1(output_dim_size_ >= input_dim_size_ - 1,
                 "Convert model error.");
    int32_t *output_dims =
        ScratchBuffer(engine_config_).GetBuffer<int32_t>(output_dim_size_);
    for (int32_t d = 0; d < static_cast<int32_t>(output_dim_size_); ++d) {
      output_dims[d] = input_dims_[d < axis_value ? d : d + 1];
    }
    ResizeOutputShape(OUTPUT, output_dim_size_, output_dims);

    int32_t outer_size = base::GetShapeSize(output_dim_size_, output_dims_);
    int32_t inner_size = input_dims_[axis_value];

    if (argmin_) {
      for (int32_t i = 0; i < outer_size; ++i) {
        int32_t idx = 0;
        T min_value = base::highest();
        const T *input_ptr = input_ + i * inner_size;
        for (int32_t j = 0; j < inner_size; ++j) {
          float input = input_ptr[j];
          if (input < min_value) {
            min_value = input;
            idx = j;
          }
        }
        output_[i] = idx;
      }
    } else {
      for (int32_t i = 0; i < outer_size; ++i) {
        int32_t idx = 0;
        T max_value = base::lowest();
        const T *input_ptr = input_ + i * inner_size;
        for (int32_t j = 0; j < inner_size; ++j) {
          float input = input_ptr[j];
          if (input > max_value) {
            max_value = input;
            idx = j;
          }
        }
        output_[i] = idx;
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int32_t axis_;
  bool keep_dims_;
  bool argmin_;

  const T *input_;
  const int32_t *input_dims_;
  uint32_t input_dim_size_;

  int32_t *output_;
  const int32_t *output_dims_;
  uint32_t output_dim_size_;

  MACE_OP_INPUT_TAGS(INPUT, AXIS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace micro


#endif  // MICRO_OPS_ARGMAX_H_
