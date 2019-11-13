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

#ifndef MICRO_OPS_STACK_H_
#define MICRO_OPS_STACK_H_

#include "micro/base/utils.h"
#include "micro/framework/operator.h"
#include "micro/framework/scratch_buffer.h"

namespace micro {
namespace ops {

template<typename T>
class StackOp : public framework::Operator {
 public:
  MaceStatus OnInit() {
    input_dims_ = GetInputShapeDims(INPUT);
    input_dim_size_ = GetInputShapeDimSize(INPUT);

    output_ = GetOutputData<T>(OUTPUT);
    axis_ = GetArgByName("axis", static_cast<int32_t>(0));

    const int32_t output_dim_size = static_cast<int32_t>(input_dim_size_) + 1;
    MACE_ASSERT1(axis_ >= -output_dim_size && axis_ < output_dim_size,
                 "axis out of bound.");
    if (axis_ < 0) {
      axis_ += output_dim_size;
    }

    return MACE_SUCCESS;
  }

  MaceStatus Run() {
    const uint32_t inputs_size = GetInputSize();
    MACE_ASSERT1(inputs_size > 0, "stack inputs are empty.");

    int32_t output_dim_size = static_cast<int32_t>(input_dim_size_) + 1;
    int32_t *output_dims =
        ScratchBuffer(engine_config_).GetBuffer<int32_t>(output_dim_size);
    for (int32_t i = 0; i < output_dim_size; ++i) {
      if (i < axis_) {
        output_dims[i] = input_dims_[i];
      } else if (i == axis_) {
        output_dims[i] = inputs_size;
      } else {
        output_dims[i] = input_dims_[i - 1];
      }
    }
    ResizeOutputShape(OUTPUT, output_dim_size, output_dims);

    int32_t high_dim_elem_size = base::accumulate_multi(input_dims_, 0, axis_);
    int32_t low_dim_elem_size =
        base::accumulate_multi(input_dims_, axis_, input_dim_size_);
    T *output_data = output_;
    for (int32_t h = 0; h < high_dim_elem_size; ++h) {
      for (uint32_t i = 0; i < inputs_size; ++i) {
        const T *input_data = GetInputData<T>(i);
        base::memcpy(output_data, input_data + h * low_dim_elem_size,
                     sizeof(T) * low_dim_elem_size);
        output_data += low_dim_elem_size;
      }
    }

    return MACE_SUCCESS;
  }

 private:
  const int32_t *input_dims_;
  uint32_t input_dim_size_;

  T *output_;

  int32_t axis_;

  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace micro

#endif  // MICRO_OPS_STACK_H_
