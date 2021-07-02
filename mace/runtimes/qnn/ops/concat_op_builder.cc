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
class ConcatOpBuilder : public OpBuilder {
 public:
  explicit ConcatOpBuilder(GraphBuilder *graph_builder)
      : OpBuilder(graph_builder) {}

  MaceStatus BuildOp(const OperatorDef &op) {
    SetOpType(QNN_OP_CONCAT);
    SetOpName(op.name().c_str());

    int axis = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(op, "axis", 3);
    const int input_dims = graph_builder_->GetTensorShape(op.input(0)).size();
    axis = axis < 0 ? axis + input_dims : axis;
    MACE_CHECK((0 <= axis && axis < input_dims),
               "Expected concatenating axis in the range [", -input_dims, ", ",
               input_dims, "], but got ", axis);
    AddScalarParam(
        QNN_OP_CONCAT_PARAM_AXIS,
        {QNN_DATATYPE_UINT_32, .uint32Value = static_cast<uint32_t>(axis)});

    MACE_CHECK(op.input_size() >= 1);
    for (int i = 0; i < op.input_size(); ++i) {
      AddInput(op.input(i));
    }
    AddOutput(op.output(0));

    return MaceStatus::MACE_SUCCESS;
  }
};
namespace qnn {
void RegisterConcat(OpRegistry *op_registry) {
  QNN_REGISTER_OP(op_registry, "Concat", ConcatOpBuilder);
}
}  // namespace qnn
}  // namespace mace
