// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#include "mace/runtimes/qnn/op_builder.h"

#include "mace/core/proto/arg_helper.h"

namespace mace {
class StridedSliceOpBuilder : public OpBuilder {
 public:
  explicit StridedSliceOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {}

  MaceStatus BuildOp(const OperatorDef &op, DataType quantized_type) {
    MACE_UNUSED(quantized_type);
    SetOpType(QNN_OP_STRIDED_SLICE);
    SetOpName(op.name().c_str());

    AddTensorParamNotCreat(QNN_OP_STRIDED_SLICE_PARAM_RANGES, op.input(3));
    AddInput(op.input(0));
    for (int i = 0; i < op.output_size(); ++i) {
      AddOutput(op.output(i));
    }

    return MaceStatus::MACE_SUCCESS;
  }
};
namespace qnn {
void RegisterStridedSlice(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "StridedSlice", StridedSliceOpBuilder);
}
}  // namespace qnn
}  // namespace mace
