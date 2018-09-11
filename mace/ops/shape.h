// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#ifndef MACE_OPS_SHAPE_H_
#define MACE_OPS_SHAPE_H_

#include <vector>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class ShapeOp : public Operator<D, T> {
 public:
  ShapeOp(const OperatorDef &op_def, OpKernelContext *context)
      : Operator<D, T>(op_def, context) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    if (input->dim_size() > 0) {
      MACE_RETURN_IF_ERROR(output->Resize({input->dim_size()}));
    } else {
      output->Resize({});
    }
    Tensor::MappingGuard output_guard(output);
    int32_t *output_data = output->mutable_data<int32_t>();

    const int data_format =
        OperatorBase::GetOptionalArg<int>("data_format", 0);
    if (input->dim_size() == 4 &&
        D == DeviceType::CPU &&
        data_format == DataFormat::NCHW) {
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
    SetFutureDefaultWaitFn(future);

    return MACE_SUCCESS;
  }

 private:
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_SHAPE_H_
