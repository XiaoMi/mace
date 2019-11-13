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

#ifndef MICRO_OPS_CAST_H_
#define MICRO_OPS_CAST_H_

#include "micro/base/utils.h"
#include "micro/base/types.h"
#include "micro/framework/operator.h"
#include "micro/include/utils/bfloat16.h"

namespace micro {
namespace ops {

#ifndef MACE_CAST_OP_CAST_TENSOR
#define MACE_CAST_OP_CAST_TENSOR(SrcType, DstType)           \
const SrcType *input = static_cast<const SrcType *>(input_); \
DstType *output = static_cast<DstType *>(output_);           \
for (int32_t i = 0; i < tensor_size_; ++i) {                 \
  output[i] = input[i];                                      \
}
#endif  // MACE_CAST_OP_CAST_TENSOR

class CastOp : public framework::Operator {
 public:
  MaceStatus OnInit() {
    input_ = GetInputData<void>(INPUT);
    input_dt_ = static_cast<DataType>(
        GetArgByName("T", static_cast<int32_t >(DT_FLOAT)));
    const int32_t *input_dims = GetInputShapeDims(INPUT);
    const uint32_t input_dim_size_ = GetInputShapeDimSize(INPUT);
    tensor_size_ = base::GetShapeSize(input_dim_size_, input_dims);
    MACE_ASSERT(tensor_size_ > 0);
    output_ = GetOutputData<void>(OUTPUT);
    output_dt_ = GetOutputDataType(OUTPUT);

    return MACE_SUCCESS;
  }

  MaceStatus Run() {
    if (input_dt_ == DT_FLOAT && output_dt_ == DT_BFLOAT16) {
#ifdef MACE_ENABLE_BFLOAT16
      MACE_CAST_OP_CAST_TENSOR(float, BFloat16)
#else
      MACE_NOT_IMPLEMENTED;
#endif
    } else if (input_dt_ == DT_BFLOAT16 && output_dt_ == DT_FLOAT) {
#ifdef MACE_ENABLE_BFLOAT16
      MACE_CAST_OP_CAST_TENSOR(BFloat16, float)
#else
      MACE_NOT_IMPLEMENTED;
#endif
    } else {
      MACE_NOT_IMPLEMENTED;
    }

    return MACE_SUCCESS;
  }

 private:
  const void *input_;
  DataType input_dt_;
  int32_t tensor_size_;

  void *output_;
  DataType output_dt_;

  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace micro


#endif  // MICRO_OPS_CAST_H_
