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

#ifndef MICRO_OPS_CONCAT_H_
#define MICRO_OPS_CONCAT_H_

#include "micro/base/utils.h"
#include "micro/framework/operator.h"
#include "micro/framework/scratch_buffer.h"

namespace micro {
namespace ops {

template <typename T>
class ConcatOp : public framework::Operator {
 public:
  MaceStatus OnInit() { return MACE_SUCCESS; }

  MaceStatus Run() {
    const int32_t *output_dims = GetOutputShapeDims(0);
    int32_t output_dim_size = GetOutputShapeDimSize(0);
    int32_t inputs_count = GetInputSize();
    MACE_ASSERT(inputs_count >= 1);

    int32_t axis = GetArgByName("axis", static_cast<int32_t>(0));
    axis = axis < 0 ? axis + output_dim_size : axis;
    MACE_ASSERT(0 <= axis && axis < output_dim_size);

    int32_t inner_size = 1;
    for (int i = 0; i < axis; ++i) {
      inner_size *= output_dims[i];
    }

    ScratchBuffer scratch_buffer(engine_config_);

    int32_t *outer_sizes = scratch_buffer.GetBuffer<int32_t>(inputs_count);
    for (int32_t i = 0; i < inputs_count; ++i) {
      const int32_t *input_dims = GetInputShapeDims(i);
      int32_t input_dim_size = GetInputShapeDimSize(i);
      MACE_ASSERT(output_dim_size == input_dim_size);

      for (int j = 0; j < output_dim_size; ++j) {
        if (j == axis) continue;
        MACE_ASSERT(input_dims[j] == output_dims[j]);
      }

      outer_sizes[i] = 1;
      for (int32_t j = axis; j < input_dim_size; ++j) {
        outer_sizes[i] *= input_dims[j];
      }
    }

    const T **input_ptrs = scratch_buffer.GetBuffer<const T *>(inputs_count);
    for (int32_t i = 0; i < inputs_count; ++i) {
      input_ptrs[i] = GetInputData<T>(i);
    }

    T *output = GetOutputData<T>(0);
    for (int32_t inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
      for (int32_t i = 0; i < inputs_count; ++i) {
        for (int32_t k = 0; k < outer_sizes[i]; ++k) {
          *output++ = *input_ptrs[i]++;
        }
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }
};

}  // namespace ops
}  // namespace micro

#endif  // MICRO_OPS_CONCAT_H_
