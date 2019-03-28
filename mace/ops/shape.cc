// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class ShapeOp : public Operation {
 public:
  explicit ShapeOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    if (input->dim_size() > 0) {
      MACE_RETURN_IF_ERROR(output->Resize({input->dim_size()}));
    } else {
      output->Resize({});
    }
    Tensor::MappingGuard output_guard(output);
    int32_t *output_data = output->mutable_data<int32_t>();

    auto has_df = Operation::GetOptionalArg<int>(
        "has_data_format", 0);
    if (has_df && input->data_format() == DataFormat::NCHW &&
        input->dim_size() != 4) {
      // transpose NCHW to NHWC for cpu runtime
      output_data[0] = static_cast<int32_t>(input->dim(0));
      output_data[1] = static_cast<int32_t>(input->dim(2));
      output_data[2] = static_cast<int32_t>(input->dim(3));
      output_data[3] = static_cast<int32_t>(input->dim(1));
    } else {
      for (unsigned int i = 0; i < input->dim_size(); ++i) {
        output_data[i] = static_cast<int32_t>(input->dim(i));
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterShape(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Shape", ShapeOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace
